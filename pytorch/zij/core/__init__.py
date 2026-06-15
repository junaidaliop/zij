# Vendored from https://github.com/pytorch/pytorch (tag v2.12.0, torch/optim/__init__.py)
# Copyright (c) Meta Platforms, Inc. and affiliates. Licensed under the BSD-3-Clause license.
# See THIRD_PARTY_NOTICES.md for details.
"""Optimizers and learning rate schedulers vendored from ``torch.optim`` at tag v2.12.0.

Most commonly used methods are already supported, and the interface is general
enough, so that more sophisticated ones can also be easily integrated in the
future.
"""

from . import lr_scheduler as lr_scheduler, swa_utils as swa_utils
from ._adafactor import Adafactor as Adafactor
from ._muon import Muon as Muon
from .adadelta import Adadelta as Adadelta
from .adagrad import Adagrad as Adagrad
from .adam import Adam as Adam
from .adamax import Adamax as Adamax
from .adamw import AdamW as AdamW
from .asgd import ASGD as ASGD
from .lbfgs import LBFGS as LBFGS
from .nadam import NAdam as NAdam
from .optimizer import Optimizer as Optimizer
from .radam import RAdam as RAdam
from .rmsprop import RMSprop as RMSprop
from .rprop import Rprop as Rprop
from .sgd import SGD as SGD
from .sparse_adam import SparseAdam as SparseAdam


del adadelta  # type: ignore[name-defined] # noqa: F821
del adagrad  # type: ignore[name-defined] # noqa: F821
del adam  # type: ignore[name-defined] # noqa: F821
del adamw  # type: ignore[name-defined] # noqa: F821
del sparse_adam  # type: ignore[name-defined] # noqa: F821
del adamax  # type: ignore[name-defined] # noqa: F821
del asgd  # type: ignore[name-defined] # noqa: F821
del sgd  # type: ignore[name-defined] # noqa: F821
del radam  # type: ignore[name-defined] # noqa: F821
del rprop  # type: ignore[name-defined] # noqa: F821
del rmsprop  # type: ignore[name-defined] # noqa: F821
del optimizer  # type: ignore[name-defined] # noqa: F821
del nadam  # type: ignore[name-defined] # noqa: F821
del lbfgs  # type: ignore[name-defined] # noqa: F821

__all__ = [
    "Adafactor",
    "Adadelta",
    "Adagrad",
    "Adam",
    "Adamax",
    "AdamW",
    "ASGD",
    "LBFGS",
    "lr_scheduler",
    "Muon",
    "NAdam",
    "Optimizer",
    "RAdam",
    "RMSprop",
    "Rprop",
    "SGD",
    "SparseAdam",
    "swa_utils",
]
