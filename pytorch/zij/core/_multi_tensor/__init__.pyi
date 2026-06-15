# Vendored from https://github.com/pytorch/pytorch (tag v2.12.0, torch/optim/_multi_tensor/__init__.pyi)
# Copyright (c) Meta Platforms, Inc. and affiliates. Licensed under the BSD-3-Clause license.
# See THIRD_PARTY_NOTICES.md for details.
from functools import partial

from ... import core as optim

Adam = partial(optim.Adam, foreach=True)
AdamW = partial(optim.AdamW, foreach=True)
NAdam = partial(optim.NAdam, foreach=True)
SGD = partial(optim.SGD, foreach=True)
RAdam = partial(optim.RAdam, foreach=True)
RMSprop = partial(optim.RMSprop, foreach=True)
Rprop = partial(optim.Rprop, foreach=True)
ASGD = partial(optim.ASGD, foreach=True)
Adamax = partial(optim.Adamax, foreach=True)
Adadelta = partial(optim.Adadelta, foreach=True)
Adagrad = partial(optim.Adagrad, foreach=True)
