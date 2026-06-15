"""``zij.optim`` — every optimizer in one namespace, mirroring ``torch.optim``.

This module re-exports the full public surface of :mod:`zij`, so both
``zij.AdamW`` and ``zij.optim.AdamW`` refer to the same class. It is imported
last by the package, after every optimizer is defined.
"""

import zij as _zij

_NAMES = [n for n in _zij.__all__ if n != "optim"]
for _name in _NAMES:
    globals()[_name] = getattr(_zij, _name)
del _name

__all__ = _NAMES
