# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa0)
# Copyright (c) 2021 Hyeongchan Kim. Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
"""Schedule-Free wrapper that makes any base optimizer schedule-free."""

from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Union

import torch

from ...core.optimizer import Optimizer

__all__ = ["ScheduleFreeWrapper"]


class ScheduleFreeWrapper:
    r"""Wraps any base optimizer to make it Schedule-Free.

    The base optimizer maintains the iterate :math:`z_t`. Gradients are
    evaluated at the interpolated point :math:`y_t`, while the parameters used
    for evaluation are the running average :math:`x_t`:

    .. math::
       \begin{aligned}
           y_t &= (1 - \beta) z_t + \beta x_t, \\
           z_{t+1} &= z_t + \text{base\_update}\bigl(\nabla f(y_t)\bigr), \\
           x_{t+1} &= (1 - c_{t+1}) x_t + c_{t+1} z_{t+1},
       \end{aligned}

    where :math:`\beta` is ``momentum`` and
    :math:`c_{t+1} = w_{t+1} / \sum_{i=1}^{t+1} w_i` with weights
    :math:`w_t = t^{\gamma} \, (\max_{i \le t} \eta_i)^{p}` for the
    learning rate :math:`\eta`, polynomial power :math:`\gamma` (``r``), and
    ``weight_lr_power`` :math:`p`. No learning rate schedule is needed.

    This memory-efficient variant swaps :math:`z_t` and :math:`y_t` in place so
    the base optimizer steps on :math:`z_t` without an extra parameter buffer.
    When using this wrapper the base optimizer's own momentum can be disabled,
    since the wrapper supplies momentum through the interpolation. Weight decay
    is applied at both :math:`z_t` (full strength) and :math:`y_t` (scaled by
    :math:`1 - \beta`).

    Reference: Aaron Defazio, Xingyu Alice Yang, Harsh Mehta, Konstantin Mishchenko,
    Ahmed Khaled, Ashok Cutkosky, "The Road Less Scheduled", NeurIPS 2024.
    https://arxiv.org/abs/2405.15682

    Note:
        Call ``optimizer.train()`` before training and ``optimizer.eval()``
        before evaluation or checkpointing, alongside the matching
        ``model.train()`` / ``model.eval()`` calls. Gradients are computed at
        :math:`y_t` while losses should be measured at :math:`x_t`, so the
        parameter buffer must be switched between the two points.

    Args:
        optimizer: base optimizer instance, or an optimizer class together with
            a ``params`` keyword argument used to construct it.
        momentum: momentum factor :math:`\beta`.
        weight_decay: weight decay (L2 penalty), applied at :math:`z_t` (full)
            and :math:`y_t` (scaled by :math:`1 - \beta`).
        r: polynomial weighting power :math:`\gamma` in the average.
        weight_lr_power: power :math:`p` for the learning-rate weighting in the
            average; 0 disables learning-rate weighting.
        maximize: maximize the objective instead of minimizing.
    """

    def __init__(
        self,
        optimizer: Union[Optimizer, type],
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        maximize: bool = False,
        **kwargs: Any,
    ):
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in the range [0.0, 1.0)")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")

        self.momentum = momentum
        self.weight_decay = weight_decay
        self.r = r
        self.weight_lr_power = weight_lr_power
        self.train_mode: bool = False
        self.maximize = maximize

        self.optimizer: Optimizer = self._load_optimizer(optimizer, **kwargs)

        self._optimizer_step_pre_hooks: Dict[int, Callable] = {}
        self._optimizer_step_post_hooks: Dict[int, Callable] = {}

        self.state: Dict = defaultdict(dict)
        self.defaults: Dict[str, Any] = self.optimizer.defaults

    @staticmethod
    def _load_optimizer(optimizer: Union[Optimizer, type], **kwargs: Any) -> Optimizer:
        if isinstance(optimizer, (Optimizer, torch.optim.Optimizer)):
            return optimizer
        if "params" in kwargs:
            params = kwargs.pop("params")
            return optimizer(params, **kwargs)
        raise ValueError("need to pass `params` when you pass the optimizer class.")

    def __str__(self) -> str:
        return "ScheduleFree"

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def __getstate__(self):
        return {"state": self.state, "optimizer": self.optimizer}

    def add_param_group(self, param_group):
        return self.optimizer.add_param_group(param_group)

    def state_dict(self) -> Dict:
        return {"schedulefree_state": self.state, "base_optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: Dict) -> None:
        self.state = state["schedulefree_state"]
        self.optimizer.load_state_dict(state["base_optimizer"])

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none)

    @torch.no_grad()
    def eval(self):
        if not self.train_mode:
            return

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "z" in state:
                    # Set p to x
                    p.lerp_(end=state["z"], weight=1.0 - 1.0 / self.momentum)

        self.train_mode = False

    @torch.no_grad()
    def train(self):
        if self.train_mode:
            return

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "z" in state:
                    # Set p to y
                    p.lerp_(end=state["z"], weight=1.0 - self.momentum)

        self.train_mode = True

    def init_group(self, group: Dict) -> None:
        if "step" not in group:
            group["step"] = 0

        for p in group["params"]:
            if p.grad is None:
                continue

            if p.grad.is_sparse:
                raise RuntimeError("ScheduleFree does not support sparse gradient.")

            state = self.state[p]
            if "z" not in state:
                state["z"] = p.clone()

    @staticmethod
    def _decouple_decay(p: torch.Tensor, lr: float, weight_decay: float, ratio: float = 1.0) -> None:
        if weight_decay > 0.0:
            p.mul_(1.0 - weight_decay * lr * ratio)

    @staticmethod
    def swap(x: torch.Tensor, y: torch.Tensor) -> None:
        x.view(torch.uint8).bitwise_xor_(y.view(torch.uint8))
        y.view(torch.uint8).bitwise_xor_(x.view(torch.uint8))
        x.view(torch.uint8).bitwise_xor_(y.view(torch.uint8))

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if not self.train_mode:
            raise ValueError("optimizer was not in train mode when step is called. call .train() before training")

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group["step"] += 1

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if self.maximize:
                    grad.neg_()

                state = self.state[p]
                z = state["z"]

                # Weight decay at z, then at y (scaled by 1 - momentum).
                self._decouple_decay(z, group["lr"], self.weight_decay)
                self._decouple_decay(p, group["lr"], self.weight_decay, ratio=1.0 - self.momentum)

                p.lerp_(end=z, weight=1.0 - 1.0 / self.momentum)

                self.swap(z, p)

        self.optimizer.step()

        for group in self.param_groups:
            lr: float = group["lr"] * group.get("d", 1.0)
            lr_max = group["lr_max"] = max(lr, group.get("lr_max", 0))

            weight: float = (group["step"] ** self.r) * (lr_max ** self.weight_lr_power)
            weight_sum = group["weight_sum"] = group.get("weight_sum", 0.0) + weight

            checkpoint: float = weight / weight_sum if weight_sum != 0.0 else 0.0

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                z = state["z"]

                self.swap(z, p)

                p.lerp_(end=z, weight=checkpoint)

                p.lerp_(end=state["z"], weight=1.0 - self.momentum)

        return loss
