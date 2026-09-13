"""Name-based registries.

Adding a dataset, an extractor, a graph builder, a relevance function or an
evaluator must be a matter of writing one class, decorating it with a name, and
writing one YAML file. No other module is edited, and in particular the
pipeline never contains an ``if name == ...`` chain.

Usage::

    from musicsim.registry import EXTRACTORS, register_extractor

    @register_extractor("mfcc")
    class MfccExtractor(Extractor):
        ...

    cls = EXTRACTORS.get("mfcc")

Registries are populated as a side effect of importing the module that defines
the entry, so :func:`load_plugins` imports the sub-packages that hold them.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator

__all__ = [
    "DATASETS",
    "EVALUATORS",
    "EXTRACTORS",
    "GRAPH_BUILDERS",
    "RELEVANCES",
    "Registry",
    "load_plugins",
    "register_dataset",
    "register_evaluator",
    "register_extractor",
    "register_graph_builder",
    "register_relevance",
]


class Registry[T]:
    """A mapping from a name to whatever was registered under it.

    The registry refuses to overwrite a name silently: registering the same
    name twice is a programming error that would make an experiment ambiguous.
    Re-registering the very same object is allowed, because importing a module
    twice under different names must not break.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, T] = {}

    def register(self, name: str, entry: T) -> T:
        """Register ``entry`` under ``name`` and return it unchanged."""
        if not name or not isinstance(name, str):
            raise ValueError(f"{self.kind} name must be a non-empty string, got {name!r}")
        previous = self._entries.get(name)
        if previous is not None and previous is not entry:
            raise KeyError(
                f"{self.kind} {name!r} is already registered as {previous!r}; "
                "names must be unique so that a configuration identifies exactly one implementation"
            )
        self._entries[name] = entry
        return entry

    def decorator(self, name: str) -> Callable[[T], T]:
        """Return a decorator that registers the decorated object under ``name``."""

        def wrap(entry: T) -> T:
            self.register(name, entry)
            return entry

        return wrap

    def get(self, name: str) -> T:
        """Look ``name`` up, with an error message that lists what is available."""
        try:
            return self._entries[name]
        except KeyError:
            available = ", ".join(self.names()) or "<none registered>"
            raise KeyError(f"unknown {self.kind} {name!r}; available: {available}") from None

    def names(self) -> list[str]:
        """Registered names, sorted."""
        return sorted(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {self.names()!r})"


DATASETS: Registry = Registry("dataset")
EXTRACTORS: Registry = Registry("extractor")
GRAPH_BUILDERS: Registry = Registry("graph builder")
RELEVANCES: Registry = Registry("relevance function")
EVALUATORS: Registry = Registry("evaluator")

register_dataset = DATASETS.decorator
register_extractor = EXTRACTORS.decorator
register_graph_builder = GRAPH_BUILDERS.decorator
register_relevance = RELEVANCES.decorator
register_evaluator = EVALUATORS.decorator

#: Sub-packages whose import populates the registries above. They are optional
#: on purpose: each one appears as the corresponding phase is implemented, and a
#: missing module must not break ``musicsim --help``.
PLUGIN_MODULES: tuple[str, ...] = (
    "musicsim.datasets",
    "musicsim.extractors",
    "musicsim.graphs",
    "musicsim.relevance",
    "musicsim.evaluation",
)


def load_plugins(modules: tuple[str, ...] = PLUGIN_MODULES) -> list[str]:
    """Import the modules that populate the registries; return the ones imported.

    Modules that do not exist yet are skipped, so the command line keeps working
    while the pipeline is being built phase by phase. Any other import error is
    a real bug and propagates.
    """
    imported: list[str] = []
    for module in modules:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            if exc.name is not None and (exc.name == module or module.startswith(f"{exc.name}.")):
                continue
            raise
        imported.append(module)
    return imported
