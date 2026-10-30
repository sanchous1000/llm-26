"""State shared by the passes of one document conversion."""

from dataclasses import dataclass, field

from source.parsing.placeholders import PlaceholderStore


@dataclass
class Context:
    """Mutable per-document state (placeholder store, warnings)."""

    base_url: str
    store: PlaceholderStore = field(default_factory=PlaceholderStore)
    warnings: list[str] = field(default_factory=list)
