# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa0)
# Copyright (c) Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# A PyTorch port of Zhang et al.'s official Lookahead (michaelrzhang/lookahead).
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the Lookahead optimizer wrapper."""

from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Union

import torch

from ...core.optimizer import Optimizer

__all__ = ["Lookahead"]


class Lookahead:
    r"""Implements Lookahead, "k steps forward, 1 step back" around any optimizer.

    Lookahead keeps two sets of weights. The fast weights :math:`\theta` are
    advanced for :math:`k` inner steps by a base optimizer, after which the slow
    weights :math:`\phi` are pulled toward them by interpolation, and the fast
    weights are reset to the slow ones:

    .. math::
       \theta_{t,0} &= \phi_{t-1} \\
       \theta_{t,i} &= \theta_{t,i-1} + A(L, \theta_{t,i-1}, d), \quad i = 1, \dots, k \\
       \phi_t &= \phi_{t-1} + \alpha\,(\theta_{t,k} - \phi_{t-1}) \\
       \theta_{t+1,0} &= \phi_t

    where :math:`A` is the inner optimizer's update on minibatch :math:`d`,
    :math:`\alpha` is the slow-weights step size, and :math:`k` is the
    synchronization period.

    Reference: Michael R. Zhang, James Lucas, Geoffrey Hinton, Jimmy Ba,
    "Lookahead Optimizer: k steps forward, 1 step back", NeurIPS 2019.
    https://arxiv.org/abs/1907.08610

    Note: this is a wrapper around a base optimizer. Pass an already constructed
    optimizer instance, e.g.
    ``Lookahead(torch.optim.Adam(model.parameters(), lr=1e-3), k=5, alpha=0.5)``.
    """

    def __init__(
        self,
        optimizer: Union[Optimizer, type],
        k: int = 5,
        alpha: float = 0.5,
        pullback_momentum: str = "none",
        **kwargs: Any,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in the range [0.0, 1.0]")
        if pullback_momentum not in ("none", "reset", "pullback"):
            raise ValueError(
                "pullback_momentum must be one of ('none', 'reset', 'pullback')"
            )

        self.optimizer: Optimizer = self.load_optimizer(optimizer, **kwargs)

        self._optimizer_step_pre_hooks: Dict[int, Callable] = {}
        self._optimizer_step_post_hooks: Dict[int, Callable] = {}

        self.alpha = alpha
        self.k = k
        self.pullback_momentum = pullback_momentum

        self.state: Dict = defaultdict(dict)

        for group in self.param_groups:
            if "counter" not in group:
                group["counter"] = 0

            for p in group["params"]:
                state = self.state[p]
                state["slow_params"] = torch.empty_like(p)
                state["slow_params"].copy_(p)
                if self.pullback_momentum == "pullback":
                    state["slow_momentum"] = torch.zeros_like(p)

        self.defaults: Dict = {
            "lookahead_alpha": alpha,
            "lookahead_k": k,
            "lookahead_pullback_momentum": pullback_momentum,
            **self.optimizer.defaults,
        }

    @staticmethod
    def load_optimizer(optimizer: Union[Optimizer, type], **kwargs: Any) -> Optimizer:
        """Return a base optimizer instance, building it from a class if needed."""
        if isinstance(optimizer, Optimizer):
            return optimizer

        if "params" in kwargs:
            params = kwargs.pop("params")
            return optimizer(params, **kwargs)

        raise ValueError("need to pass `params` when you pass the optimizer class.")

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def __getstate__(self):
        return {
            "state": self.state,
            "optimizer": self.optimizer,
            "alpha": self.alpha,
            "k": self.k,
            "pullback_momentum": self.pullback_momentum,
        }

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def backup_and_load_cache(self) -> None:
        r"""Back up the fast weights and load the slow weights for evaluation."""
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["backup_params"] = torch.empty_like(p)
                state["backup_params"].copy_(p)
                p.data.copy_(state["slow_params"])

    def clear_and_load_backup(self) -> None:
        r"""Restore the fast weights that were saved by ``backup_and_load_cache``."""
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                p.data.copy_(state["backup_params"])
                del state["backup_params"]

    def state_dict(self) -> Dict:
        lookahead_state: Dict = {
            p: dict(param_state) for p, param_state in self.state.items()
        }
        return {
            "lookahead_state": lookahead_state,
            "base_optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: Dict) -> None:
        r"""Load state."""
        lookahead_state = state["lookahead_state"]
        self.state = defaultdict(
            dict, {p: dict(param_state) for p, param_state in lookahead_state.items()}
        )
        self.optimizer.load_state_dict(state["base_optimizer"])

    @torch.no_grad()
    def update(self, group: Dict) -> None:
        for p in group["params"]:
            if p.grad is None:
                continue

            state = self.state[p]

            slow = state["slow_params"]

            p.mul_(self.alpha).add_(slow, alpha=1.0 - self.alpha)
            slow.copy_(p)

            if "momentum_buffer" not in self.optimizer.state[p]:
                self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p)

            if self.pullback_momentum == "pullback":
                internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(
                    self.alpha
                ).add_(state["slow_momentum"], alpha=1.0 - self.alpha)
                state["slow_momentum"] = self.optimizer.state[p]["momentum_buffer"]
            elif self.pullback_momentum == "reset":
                self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p)

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
                self.update(group)
        return loss
