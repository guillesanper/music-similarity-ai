"""MERT-v1-95M (Li et al. 2023, arXiv:2306.00107): a frozen HuBERT-style
transformer pre-trained on 24 kHz music audio. A forward pass returns 13
hidden states -- the convolutional feature extractor's output plus 12
transformer layers, 768 dimensions each -- and which one performs best is an
empirical question (the model card says so explicitly), so this extractor
keeps all 13 and reduces to a single "titular" layer only at the very end,
exactly like every other extractor's ``(dim,)`` contract.

Two things make this extractor different from :mod:`musicsim.extractors.mfcc`:

1. **Windowing.** MERT was pre-trained on 5 s crops, not 30 s clips (see its
   model card). Self-attention cost grows with the square of the sequence
   length, so one 30 s forward pass would cost roughly 36x more than six 5 s
   ones for the same audio -- this extractor always runs the shorter windows,
   mean-pooling their per-layer vectors afterwards. It is the same "many short
   crops, then averaged" strategy phase S3's CNN uses at 3 s.
2. **Two-level caching.** All 13 layers are cached per item, independently of
   ``extractor.mert_layer`` (:func:`load_or_compute_mert_layers`), so that
   choosing a different titular layer -- the validation-only selection in
   :func:`select_titular_layer`, C.10 "selección de modelo" -- never pays for
   a second forward pass through the model. :meth:`MertExtractor.extract_one`
   reads that cache and slices out the titular layer, which is the only thing
   ``extract_stage``'s generic ``(N, dim)`` cache ever sees.

3. **Device.** ``extractor.device`` (``auto`` by default) runs the model on a
   CUDA GPU when there is one and on the CPU otherwise. The CPU path is the
   original one: one 5 s window per forward, one process, one item at a time.
   On a GPU, windows from consecutive items are grouped into batches of
   ``extractor.batch_windows`` and decoded ahead of time on a few threads
   (:meth:`MertExtractor._extract_all_batched`), because decoding 30 s of mp3
   costs about as much as the forward pass itself. Device, batch size,
   precision and thread counts are execution parameters, like ``n_jobs``:
   none of them is part of the layer-cache key, so a cache filled on the CPU
   is reused as is on a GPU (the values agree to float tolerance, not bit for
   bit).

The weights (``extractor.model_name``, pinned to ``extractor.revision``) are
downloaded from Hugging Face the first time they are needed and live under the
user's ``~/.cache/huggingface`` -- never inside this repository, exactly like
every other cache this project writes. They are released under CC-BY-NC-4.0
(non-commercial research use), which this thesis is; see
``docs/sources.md`` and ``report/sections/04_methods.tex``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from musicsim import audio, paths
from musicsim.config import Config, ConfigError, config_hash
from musicsim.extractors.base import Extractor
from musicsim.registry import register_extractor

__all__ = ["MertExtractor", "load_or_compute_mert_layers", "select_titular_layer"]

_LOG = logging.getLogger(__name__)

_PRECISIONS = ("float32", "float16")


def _resolve_device(config: Config) -> str:
    """``"cpu"`` or a CUDA device string, from ``extractor.device``.

    ``auto`` picks CUDA when torch sees a GPU and the CPU otherwise (also when
    torch is not installed at all; the missing-extra error then surfaces when
    the model is first needed, as it always has). An explicit ``cuda`` that
    cannot be honoured is an error rather than a silent fallback.
    """
    requested = str(config.get("extractor.device", "auto")).strip().lower()
    if requested == "cpu":
        return "cpu"
    if requested != "auto" and not re.fullmatch(r"cuda(:\d+)?", requested):
        raise ConfigError(
            f"extractor.device must be 'auto', 'cpu', 'cuda' or 'cuda:N', got {requested!r}"
        )
    try:
        import torch
    except ImportError:
        if requested == "auto":
            return "cpu"
        raise ConfigError(
            f"extractor.device={requested!r} needs torch; install it with `pip install -e '.[dl]'`"
        ) from None
    if torch.cuda.is_available():
        return "cuda" if requested == "auto" else requested
    if requested == "auto":
        return "cpu"
    raise ConfigError(f"extractor.device={requested!r} but torch reports no CUDA device")


def _positive_int(config: Config, key: str, default: int) -> int:
    value = int(config.get(key, default))
    if value < 1:
        raise ConfigError(f"{key} must be at least 1, got {value}")
    return value


def _is_oom(exc: BaseException) -> bool:
    """Whether ``exc`` is a CUDA out-of-memory error (a ``RuntimeError`` subclass
    in torch; matched by message so it needs no torch import).
    """
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


@register_extractor("mert")
class MertExtractor(Extractor):
    """Reduces MERT's 13 hidden states to the configured titular layer.

    ``torch``/``transformers`` are imported lazily (inside methods, never at
    module scope) so that importing this module -- which happens on every
    ``musicsim`` invocation through :func:`musicsim.registry.load_plugins` --
    does not require the optional ``[dl]`` extra. The error only surfaces when
    MERT is actually asked to extract something.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._model = None
        self._processor = None
        self._device: str | None = None
        # Largest forward batch that has not run out of memory yet (None until
        # an OOM teaches us a limit).
        self._oom_limit: int | None = None

    def extract_one(self, item_id: int, path: str) -> np.ndarray:
        return self._titular(load_or_compute_mert_layers(self, item_id, path))

    def _titular(self, layers: np.ndarray) -> np.ndarray:
        layer = int(self.config.require("extractor.mert_layer"))
        n_layers = layers.shape[0]
        if not 0 <= layer < n_layers:
            raise ValueError(f"extractor.mert_layer must be in [0, {n_layers}), got {layer}")
        return layers[layer].astype(np.float32)

    @property
    def device(self) -> str:
        """The resolved device, ``"cpu"`` or a CUDA device string."""
        if self._device is None:
            self._device = _resolve_device(self.config)
        return self._device

    def extract_all(self, index: pd.DataFrame, *, n_jobs: int = -1, progress: bool = True):
        """Always single-process, whatever ``runtime.n_jobs`` says.

        The model (~95M parameters) is loaded once per process. On the CPU it
        already parallelises its own matrix multiplications through BLAS/MKL
        threads, so spreading items across worker processes would reload the
        model in each one and have them contend for the same cores. On a GPU
        one process owns the device and its memory, so more processes would
        only reload the weights and compete for VRAM; parallelism there comes
        from batching and from decoding audio on threads.
        """
        if self.device == "cpu":
            return super().extract_all(index, n_jobs=1, progress=progress)
        return self._extract_all_batched(index, progress=progress)

    # -- model loading -----------------------------------------------------
    def _precision(self) -> str:
        precision = str(self.config.get("extractor.precision", "float32")).strip().lower()
        if precision not in _PRECISIONS:
            raise ConfigError(
                f"extractor.precision must be one of {_PRECISIONS}, got {precision!r}"
            )
        return precision

    def _model_and_target_sr(self):
        if self._model is None:
            import torch
            from transformers import AutoModel, Wav2Vec2FeatureExtractor

            model_name = self.config.require("extractor.model_name")
            revision = self.config.get("extractor.revision")
            trust = bool(self.config.get("extractor.trust_remote_code", True))
            device = self.device
            precision = self._precision()

            model = AutoModel.from_pretrained(
                model_name, revision=revision, trust_remote_code=trust
            )
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            torch.set_grad_enabled(False)
            if device == "cpu":
                # extract_all already pins this extractor to one process (see
                # above): let that one process use every core for its matrix
                # multiplications instead of torch's conservative default. On a
                # GPU the cores are left to the threads that decode audio.
                torch.set_num_threads(os.cpu_count() or 1)
            else:
                model.to(device)

            processor = Wav2Vec2FeatureExtractor.from_pretrained(
                model_name, revision=revision, trust_remote_code=trust
            )
            self._model = model
            self._processor = processor
            _LOG.info("MERT on %s (precision=%s)", device, precision if device != "cpu" else "n/a")
        return self._model, int(self._processor.sampling_rate)

    def _ensure_loaded(self) -> None:
        """Load the model now. Called from the main thread before any decoding
        thread starts, so the threads never race to load it.
        """
        self._model_and_target_sr()

    # -- audio -> windows ----------------------------------------------------
    def _load_windows(self, path: str) -> np.ndarray:
        """Decode ``path`` to ``(n_windows, window_samples)`` float32 at the
        model's sampling rate: non-overlapping ``window_seconds`` windows, the
        trailing remainder dropped. Pure CPU/IO work, safe to run on threads.
        """
        import librosa

        _, target_sr = self._model_and_target_sr()
        y = audio.load_clip(path, self.config)
        pipeline_sr = int(self.config.require("audio.sample_rate"))
        if pipeline_sr != target_sr:
            y = librosa.resample(y, orig_sr=pipeline_sr, target_sr=target_sr)

        window_seconds = float(self.config.get("extractor.window_seconds", 5.0))
        window_samples = round(window_seconds * target_sr)
        n_windows = max(1, len(y) // window_samples)
        y = y[: n_windows * window_samples]
        return np.ascontiguousarray(y.reshape(n_windows, window_samples), dtype=np.float32)

    # -- windows -> pooled hidden states ---------------------------------------
    def _forward_pooled(self, batch: np.ndarray) -> np.ndarray:
        """One forward pass over ``(B, window_samples)`` windows: ``(B, n_layers,
        hidden_size)`` float32, each layer mean-pooled over time. The pooling
        happens on the device so that only ``B * 13 * 768`` numbers come back.
        """
        import torch

        model, _ = self._model_and_target_sr()
        inputs = torch.from_numpy(np.ascontiguousarray(batch)).float().to(self.device)
        use_half = self._precision() == "float16" and self.device != "cpu"
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_half):
            outputs = model(inputs, output_hidden_states=True)
        pooled = torch.stack([hidden.float().mean(dim=1) for hidden in outputs.hidden_states], 1)
        return pooled.cpu().numpy()

    def _release_cuda_cache(self) -> None:
        with contextlib.suppress(ImportError):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _forward_chunked(self, windows: np.ndarray) -> np.ndarray:
        """``(n, n_layers, hidden_size)`` float32 for ``(n, window_samples)``
        windows, run through the model in chunks. On the CPU a chunk is one
        window (grouping them did not help there); on a GPU it is up to
        ``extractor.batch_windows``, halved for good the first time a forward
        runs out of memory -- a batch size that was too optimistic costs speed,
        not the run.
        """
        n_layers = int(self.config.get("extractor.n_layers", 13))
        size = 1
        if self.device != "cpu":
            size = _positive_int(self.config, "extractor.batch_windows", 24)
        if self._oom_limit is not None:
            size = min(size, self._oom_limit)

        out: list[np.ndarray] = []
        start = 0
        while start < len(windows):
            chunk = windows[start : start + size]
            out_of_memory = False
            try:
                pooled = self._forward_pooled(chunk)
            except RuntimeError as exc:
                if size == 1 or not _is_oom(exc):
                    raise
                out_of_memory = True
            if out_of_memory:
                size //= 2
                self._oom_limit = size
                _LOG.warning("MERT forward ran out of memory; retrying with %d window(s)", size)
                self._release_cuda_cache()
                continue
            if pooled.shape[1] != n_layers:
                raise ValueError(
                    f"model returned {pooled.shape[1]} hidden states, "
                    f"extractor.n_layers is {n_layers}"
                )
            out.append(pooled)
            start += len(chunk)

        hidden_size = int(self.config.get("extractor.hidden_size", 768))
        if not out:
            return np.empty((0, n_layers, hidden_size), dtype=np.float32)
        return np.concatenate(out, axis=0)

    def compute_layers(self, path: str) -> np.ndarray:
        """Forward pass: ``(n_layers, hidden_size)`` float16, mean-pooled over
        time within each of the non-overlapping ``window_seconds`` windows and
        then averaged across windows.
        """
        windows = self._load_windows(path)
        return self._forward_chunked(windows).mean(axis=0).astype(np.float16)

    # -- batched extraction (GPU) --------------------------------------------
    def _extract_all_batched(self, index: pd.DataFrame, *, progress: bool = True):
        """Same contract as :meth:`Extractor.extract_all`, built for a GPU.

        Items that already have their 13 layers cached are read from disk and
        never touch audio or the device. The rest are decoded on
        ``extractor.decode_workers`` threads, at most ``extractor.prefetch``
        items ahead of the consumer (a bounded sliding window: decoded audio is
        ~3 MB per item, so an unbounded read-ahead over the whole index would
        hold tens of GB). The consumer groups the windows of consecutive items
        until it has ``extractor.batch_windows`` of them, runs one forward, and
        writes each item's cache file as soon as its batch is done, so an
        interrupted run resumes from the cache. A file that cannot be decoded
        is recorded as a failure and never reaches the GPU.
        """
        dataset = self.config.require("dataset.directory")
        cache = bool(self.config.get("runtime.cache", True))
        cache_dir = _mert_layers_cache_dir(dataset, self.config, create=cache)

        n_layers = int(self.config.get("extractor.n_layers", 13))
        layer = int(self.config.require("extractor.mert_layer"))
        if not 0 <= layer < n_layers:
            raise ValueError(f"extractor.mert_layer must be in [0, {n_layers}), got {layer}")
        batch_windows = _positive_int(self.config, "extractor.batch_windows", 24)
        workers = _positive_int(self.config, "extractor.decode_workers", 4)
        prefetch = _positive_int(self.config, "extractor.prefetch", 8)
        self._precision()  # reject a bad value before any work is done

        item_ids = [int(i) for i in index["item_id"]]
        audio_paths = [str(p) for p in index["path"]]
        n = len(item_ids)
        vectors: list[np.ndarray | None] = [None] * n
        failures: dict[int, str] = {}
        bar = tqdm(total=n, desc=self.name, unit="item", disable=not progress)

        def finish(pos: int, layers: np.ndarray) -> None:
            vector = self._titular(layers)
            if not np.isfinite(vector).all():
                raise ValueError("vector contains NaN or inf")
            vectors[pos] = vector

        def cache_path(pos: int) -> Path:
            return cache_dir / f"{item_ids[pos]:06d}.npy"

        todo: list[int] = []
        for pos in range(n):
            if cache and cache_path(pos).is_file():
                try:
                    finish(pos, np.load(cache_path(pos)))
                except Exception as exc:
                    failures[pos] = f"{type(exc).__name__}: {exc}"
                bar.update(1)
            else:
                todo.append(pos)

        batch: list[tuple[int, np.ndarray]] = []

        def flush() -> None:
            if not batch:
                return
            pooled = self._forward_chunked(np.concatenate([w for _, w in batch]))
            offset = 0
            for pos, windows in batch:
                layers = pooled[offset : offset + len(windows)].mean(axis=0).astype(np.float16)
                offset += len(windows)
                try:
                    if cache:
                        _save_layers(cache_path(pos), layers)
                    finish(pos, layers)
                except Exception as exc:
                    failures[pos] = f"{type(exc).__name__}: {exc}"
                bar.update(1)
            batch.clear()

        try:
            if todo:
                self._ensure_loaded()
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    pending: deque = deque()
                    upcoming = iter(todo)

                    def top_up() -> None:
                        while len(pending) < prefetch:
                            pos = next(upcoming, None)
                            if pos is None:
                                return
                            pending.append((pos, pool.submit(self._load_windows, audio_paths[pos])))

                    top_up()
                    n_windows = 0
                    while pending:
                        pos, future = pending.popleft()
                        top_up()
                        try:
                            windows = future.result()
                        except Exception as exc:
                            failures[pos] = f"{type(exc).__name__}: {exc}"
                            bar.update(1)
                            continue
                        batch.append((pos, windows))
                        n_windows += len(windows)
                        if n_windows >= batch_windows:
                            flush()
                            n_windows = 0
                    flush()
        finally:
            bar.close()

        ok = [pos for pos in range(n) if vectors[pos] is not None]
        if ok:
            X = np.stack([vectors[pos] for pos in ok]).astype(np.float32, copy=False)
        else:
            X = np.empty((0, self.dim), np.float32)
        ids = np.asarray([item_ids[pos] for pos in ok], dtype=np.int64)
        failed = [(item_ids[pos], failures[pos]) for pos in sorted(failures)]
        return X, ids, failed


def _mert_layers_cache_dir(dataset: str, config: Config, *, create: bool = False) -> Path:
    """Cache directory for the raw 13-layer arrays, keyed independently of
    ``extractor.mert_layer``/``extractor.dim`` so that choosing a different
    titular layer never invalidates it (see the module docstring).
    """
    resolved = config.to_dict()
    extractor_cfg = resolved.get("extractor", {})
    key_cfg = {
        "audio": resolved.get("audio", {}),
        "extractor": {
            name: extractor_cfg.get(name)
            for name in (
                "name",
                "model_name",
                "revision",
                "window_seconds",
                "n_layers",
                "hidden_size",
            )
        },
    }
    key = config_hash(key_cfg)
    return paths.cache_dir(dataset, "mert-layers", key, create=create)


def load_or_compute_mert_layers(extractor: MertExtractor, item_id: int, path: str) -> np.ndarray:
    """``(n_layers, hidden_size)`` float16, cached on disk per item.

    Mirrors :func:`musicsim.audio.load_or_compute_melspec`: the expensive
    forward pass runs at most once per item per (audio, model) configuration,
    shared by every ``extractor.mert_layer`` choice.
    """
    config = extractor.config
    dataset = config.require("dataset.directory")
    cache = bool(config.get("runtime.cache", True))
    cache_dir = _mert_layers_cache_dir(dataset, config, create=cache)
    cache_path = cache_dir / f"{int(item_id):06d}.npy"

    if cache and cache_path.is_file():
        return np.load(cache_path)

    layers = extractor.compute_layers(path)
    if cache:
        _save_layers(cache_path, layers)
    return layers


def _save_layers(cache_path: Path, layers: np.ndarray) -> None:
    """Write ``layers`` so that ``cache_path`` is either absent or complete.

    A run interrupted halfway through ``np.save`` would otherwise leave a
    truncated file that counts as "already computed" and fails to load later.
    """
    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    with open(tmp_path, "wb") as handle:
        np.save(handle, layers)
    os.replace(tmp_path, cache_path)


def select_titular_layer(
    layers: np.ndarray,
    ids: np.ndarray,
    index: pd.DataFrame,
    *,
    k: int = 10,
) -> tuple[int, np.ndarray]:
    """Pick the layer with the highest Recall@``k`` on FMA **validation**.

    ``layers`` is ``(N, n_layers, hidden_size)``, row-aligned with ``ids``;
    ``index`` is the dataset's item index (``item_id``, ``split``,
    ``genre_top``). Only rows whose ``split`` is ``"validation"`` ever enter
    the computation -- ``training`` and, crucially, ``test`` rows are read out
    of ``index`` but never ranked or scored (C.10 "selección de modelo": model
    selection happens on validation, never on test).

    Standardisation and PCA are skipped on purpose: every layer is compared
    under the same bare L2-normalisation, which is what makes the comparison
    fair between layers. The titular layer, once chosen, goes through the
    real ``embedding.*`` pipeline like any other representation.

    Returns ``(layer, r_at_k)``: the winning layer index and each layer's mean
    Recall@``k`` (the ``dedup`` variant), for the per-layer figure.
    """
    from musicsim.evaluation.retrieval import compute_per_query_metrics
    from musicsim.graphs.duplicates import duplicate_groups, near_duplicate_pairs
    from musicsim.relevance.base import label_codes

    if layers.shape[0] != len(ids):
        raise ValueError(f"layers has {layers.shape[0]} rows but {len(ids)} ids were given")

    index_by_id = index.set_index("item_id")
    missing = pd.Index(ids).difference(index_by_id.index)
    if len(missing):
        raise ValueError(f"{len(missing)} id(s) not present in the index, e.g. {list(missing[:5])}")

    split = index_by_id.loc[ids, "split"].to_numpy()
    val_mask = split == "validation"
    if not val_mask.any():
        raise ValueError("no validation-split item in `ids`; nothing to select a layer on")

    val_ids = np.asarray(ids)[val_mask]
    val_layers = layers[val_mask].astype(np.float32)
    val_index = index_by_id.loc[val_ids]
    genre_codes = label_codes(val_index["genre_top"])
    rows = np.arange(len(val_ids))

    n_layers = val_layers.shape[1]
    scores = np.empty(n_layers, dtype=np.float64)
    for layer in range(n_layers):
        X = val_layers[:, layer, :]
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.where(norms == 0, 1.0, norms)
        pairs = near_duplicate_pairs(X, val_ids)
        groups = duplicate_groups(val_ids, pairs)
        per_query = compute_per_query_metrics(X, val_ids, rows, groups, genre_codes, ks=[k])
        scores[layer] = per_query.loc[per_query["metric"] == "R", "value"].mean()

    return int(np.argmax(scores)), scores
