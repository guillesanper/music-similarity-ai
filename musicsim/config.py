"""Experiment configuration: YAML composition, overrides, validation and hashing.

Every experiment is one YAML file. A file may inherit from others through a
``defaults:`` list, exactly like this::

    # configs/experiments/e01_mfcc_baseline.yaml
    defaults:
      - ../base.yaml
      - ../datasets/fma_small.yaml
      - ../extractors/mfcc.yaml
    experiment: e01_mfcc_baseline
    retrieval:
      queries: all

Composition rules
-----------------
* Entries of ``defaults`` are resolved relative to the file that lists them and
  merged left to right; the including file wins over all of them.
* Mappings are merged key by key (deep merge). Any other value, lists included,
  replaces the inherited one outright: a list in a configuration is a single
  decision ("these k values"), never something to append to.
* Each file is loaded once per composition, so a diamond inheritance does not
  apply the same file twice, and a cycle is reported instead of hanging.

Overrides
---------
``key.subkey=value`` strings coming from the command line are applied after the
composition. Values are parsed as YAML, so ``seed=0``, ``pca.enabled=true`` and
``retrieval.ks=[1,5,10]`` all do what they look like.

Hashing
-------
:func:`config_hash` is stable across processes and machines: it serialises the
resolved configuration with sorted keys and hashes the bytes with SHA-256. It is
what names the cache directories and the run directories, so two runs share a
directory if and only if they ran the same configuration.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "Config",
    "ConfigError",
    "apply_overrides",
    "compose",
    "config_hash",
    "deep_merge",
    "load_config",
    "parse_override",
]

#: Key holding the inheritance list.
DEFAULTS_KEY = "defaults"


class ConfigError(ValueError):
    """A configuration is missing, malformed, or asks for something impossible."""


@dataclasses.dataclass(frozen=True, slots=True)
class Config:
    """A resolved configuration: a frozen view over a plain nested mapping.

    The mapping is kept as data rather than mirrored into one dataclass per
    section, because every phase adds sections and a rigid schema would have to
    be edited for each one. What is frozen is the object: :meth:`get` and
    :meth:`require` read, and nothing writes.
    """

    data: Mapping[str, Any]
    source: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, Mapping):
            raise ConfigError(f"a configuration must be a mapping, got {type(self.data).__name__}")
        object.__setattr__(self, "data", _freeze(self.data))

    # -- reading ---------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        """Value at a dotted path, or ``default`` when any segment is missing."""
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        """Value at a dotted path; raise :class:`ConfigError` when it is missing."""
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            where = f" in {self.source}" if self.source else ""
            raise ConfigError(f"required configuration key {dotted!r} is missing{where}")
        return value

    def section(self, name: str) -> Mapping[str, Any]:
        """A sub-mapping, or an empty one when the section is absent."""
        value = self.get(name, {})
        if not isinstance(value, Mapping):
            raise ConfigError(f"configuration section {name!r} must be a mapping")
        return value

    def __contains__(self, dotted: object) -> bool:
        sentinel = object()
        return isinstance(dotted, str) and self.get(dotted, sentinel) is not sentinel

    # -- identity --------------------------------------------------------
    @property
    def name(self) -> str:
        """Experiment name; falls back to the file stem, then to ``unnamed``."""
        value = self.get("experiment")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return self.source.stem if self.source is not None else "unnamed"

    def hash(self) -> str:
        """Stable SHA-256 of the resolved configuration."""
        return config_hash(self.data)

    def to_dict(self) -> dict[str, Any]:
        """A mutable deep copy, for serialisation."""
        return _thaw(self.data)

    def to_yaml(self) -> str:
        """The configuration as YAML with sorted keys, ready to be written out."""
        return yaml.safe_dump(self.to_dict(), sort_keys=True, allow_unicode=True, indent=2)

    def write(self, path: Path) -> Path:
        """Write :meth:`to_yaml` to ``path`` (this is ``config.resolved.yaml``)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml(), encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Loading and composition
# ---------------------------------------------------------------------------
def load_config(path: Path | str, overrides: Sequence[str] = ()) -> Config:
    """Compose ``path`` with its ``defaults`` chain and apply ``overrides``."""
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise ConfigError(f"configuration file not found: {resolved}")
    data = compose(resolved)
    if overrides:
        data = apply_overrides(data, overrides)
    return Config(data=data, source=resolved.resolve())


def compose(path: Path | str) -> dict[str, Any]:
    """Read a YAML file and merge everything its ``defaults`` chain pulls in."""
    return _compose(Path(path).resolve(), stack=(), seen=set())


def _compose(path: Path, stack: tuple[Path, ...], seen: set[Path]) -> dict[str, Any]:
    if path in stack:
        chain = " -> ".join(p.name for p in (*stack, path))
        raise ConfigError(f"circular defaults in configuration: {chain}")
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path} must contain a mapping at the top level")

    node = dict(raw)
    parents = node.pop(DEFAULTS_KEY, [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, Sequence):
        raise ConfigError(f"{path}: {DEFAULTS_KEY!r} must be a string or a list of strings")

    merged: dict[str, Any] = {}
    for parent in parents:
        if not isinstance(parent, str):
            raise ConfigError(f"{path}: every entry of {DEFAULTS_KEY!r} must be a string")
        parent_path = (path.parent / parent).resolve()
        if parent_path in seen:
            # Already merged through another branch of a diamond: skip it so
            # that it is not applied twice.
            continue
        seen.add(parent_path)
        merged = deep_merge(merged, _compose(parent_path, (*stack, path), seen))

    return deep_merge(merged, node)


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``update`` onto ``base``; mappings recurse, everything else replaces."""
    result = dict(base)
    for key, value in update.items():
        current = result.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(current, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------
def parse_override(text: str) -> tuple[list[str], Any]:
    """Split ``a.b=value`` into its key path and its YAML-parsed value."""
    if "=" not in text:
        raise ConfigError(f"override {text!r} must have the form key=value or key.subkey=value")
    key, _, raw = text.partition("=")
    key = key.strip()
    if not key:
        raise ConfigError(f"override {text!r} has an empty key")
    parts = key.split(".")
    if any(not part for part in parts):
        raise ConfigError(f"override key {key!r} has an empty path segment")
    try:
        value = yaml.safe_load(raw) if raw.strip() else ""
    except yaml.YAMLError as exc:
        raise ConfigError(f"override {text!r}: value is not valid YAML ({exc})") from exc
    return parts, value


def apply_overrides(data: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Return ``data`` with every ``key=value`` override applied."""
    result = _thaw(data)
    for text in overrides:
        parts, value = parse_override(text)
        node = result
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value
    return result


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def config_hash(data: Mapping[str, Any]) -> str:
    """Stable SHA-256 hex digest of a configuration.

    Stability matters more than speed here: the digest names cache and run
    directories, so it must not depend on the insertion order of the keys, on
    the Python version, or on ``PYTHONHASHSEED``. Keys are sorted and the
    payload is UTF-8 JSON; values JSON cannot represent are rendered with
    ``repr`` rather than silently dropped.
    """
    payload = json.dumps(
        _thaw(data),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=repr,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Freezing helpers
# ---------------------------------------------------------------------------
class _FrozenDict(Mapping):
    """A read-only mapping, so that a Config cannot be mutated by accident."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = dict(data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data!r})"


def _freeze(value: Any) -> Any:
    """Recursively turn mappings into read-only mappings and lists into tuples."""
    if isinstance(value, Mapping):
        return _FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    """Inverse of :func:`_freeze`: plain dicts and lists, deep-copied."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)
