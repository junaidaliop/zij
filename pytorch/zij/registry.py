"""Name-based optimizer lookup and discovery."""

from fnmatch import fnmatch

_REGISTRY: dict[str, type] = {}
# How each optimizer is constructed: "params" (a parameter iterable, the common
# case), "model" (a torch.nn.Module, e.g. Adam-mini and LOMO read parameter
# names), or "wrapper" (wraps a base optimizer plus block structure, e.g. BAdam).
_CONSTRUCTION: dict[str, str] = {}


def _register(cls: type, name: str | None = None, construction: str = "params") -> type:
    key = (name or cls.__name__).lower()
    existing = _REGISTRY.get(key)
    if existing is not None and existing is not cls:
        raise ValueError(f"optimizer name already registered: {key}")
    _REGISTRY[key] = cls
    _CONSTRUCTION[key] = construction
    return cls


def load_optimizer(name: str) -> type:
    """Return the optimizer class registered under ``name`` (case-insensitive).

    Example:
        >>> opt_cls = load_optimizer("adamw")
        >>> optimizer = opt_cls(model.parameters(), lr=1e-3)
    """
    key = name.lower()
    try:
        return _REGISTRY[key]
    except KeyError:
        raise ValueError(
            f"unknown optimizer: {name!r}; see list_optimizers()"
        ) from None


def construction_of(name: str) -> str:
    """Return how an optimizer is constructed: "params", "model", or "wrapper"."""
    return _CONSTRUCTION[name.lower()]


def list_optimizers(pattern: str = "*") -> list[str]:
    """Return the sorted registered optimizer names matching a wildcard pattern.

    Example:
        >>> list_optimizers("adam*")
        ['adam', 'adamax', 'adamw']
    """
    return sorted(k for k in _REGISTRY if fnmatch(k, pattern.lower()))
