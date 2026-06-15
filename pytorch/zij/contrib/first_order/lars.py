# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa0)
# Copyright (c) Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the LARS optimizer."""

from typing import Callable, List, Optional

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["LARS"]


class LARS(Optimizer):
    r"""Implements LARS, layer-wise adaptive rate scaling for large-batch SGD.

    Each multidimensional parameter (treated as a layer) is given a local
    learning rate proportional to the ratio between the norm of its weights and
    the norm of its gradient, so the magnitude of every layer's update no longer
    depends on the scale of its gradient:

    .. math::
       \begin{aligned}
       \lambda^l &= \eta \, \frac{\lVert \theta^l \rVert}
                                  {\lVert g^l \rVert}                          \\
       g^l       &\leftarrow \lambda^l \left( g^l + \beta\, \theta^l \right)   \\
       v_t^l     &= \mu\, v_{t-1}^l + g^l                                      \\
       \theta_t^l &= \theta_{t-1}^l - \gamma\, v_t^l
       \end{aligned}

    where :math:`\eta` is the trust coefficient, :math:`\beta` the weight decay,
    :math:`\mu` the momentum, and :math:`\gamma` the global learning rate. The
    trust ratio :math:`\lambda^l` falls back to one when either norm is zero.
    Parameters with one dimension or fewer (biases and scalars) skip the rate
    scaling and weight decay and are updated by plain momentum SGD.

    Reference: Yang You, Igor Gitman, Boris Ginsburg, "Large Batch Training of
    Convolutional Networks", arXiv 2017.
    https://arxiv.org/abs/1708.03888

    Note:
        ``foreach`` selects a multi-tensor implementation that yields the same
        result as the per-parameter path; it is disabled for groups using
        Nesterov momentum.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        dampening: float = 0.0,
        trust_coefficient: float = 1e-3,
        nesterov: bool = False,
        foreach: Optional[bool] = None,
        maximize: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if weight_decay < 0.0:
            raise ValueError(
                f"Invalid weight_decay value: {weight_decay} - should be >= 0.0"
            )
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(
                f"Invalid momentum value: {momentum} - should be in [0.0, 1.0]"
            )
        if not 0.0 <= dampening <= 1.0:
            raise ValueError(
                f"Invalid dampening value: {dampening} - should be in [0.0, 1.0]"
            )
        if trust_coefficient < 0.0:
            raise ValueError(
                f"Invalid trust_coefficient value: {trust_coefficient}"
                " - should be >= 0.0"
            )

        self.foreach = foreach
        self.maximize = maximize

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "dampening": dampening,
            "trust_coefficient": trust_coefficient,
            "nesterov": nesterov,
            "foreach": foreach,
        }
        super().__init__(params, defaults)

    def _init_group(self, group) -> None:
        if "step" not in group:
            group["step"] = 0

        for p in group["params"]:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError("LARS does not support sparse gradients")

            if group["momentum"] > 0.0:
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = p.grad.clone()

    def _can_use_foreach(self, group) -> bool:
        if group.get("foreach") is False or group["nesterov"]:
            return False
        return True

    def _step_foreach(
        self,
        group,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        momentum_buffers: List[torch.Tensor],
    ) -> None:
        if self.maximize:
            torch._foreach_neg_(grads)

        masks = [p.ndim > 1 for p in params]
        masked_params = [p for p, m in zip(params, masks) if m]
        masked_grads = [g for g, m in zip(grads, masks) if m]

        if masked_params:
            param_norms = torch._foreach_norm(masked_params)
            grad_norms = torch._foreach_norm(masked_grads)

            trust_ratios = []
            for pn, gn in zip(param_norms, grad_norms):
                one = torch.ones_like(pn)
                trust_ratio = torch.where(
                    pn > 0.0,
                    torch.where(
                        gn > 0.0, group["trust_coefficient"] * pn / gn, one
                    ),
                    one,
                )
                trust_ratios.append(trust_ratio)

            torch._foreach_add_(
                masked_grads, masked_params, alpha=group["weight_decay"]
            )
            torch._foreach_mul_(masked_grads, trust_ratios)

        torch._foreach_mul_(momentum_buffers, group["momentum"])
        torch._foreach_add_(
            momentum_buffers, grads, alpha=1.0 - group["dampening"]
        )
        torch._foreach_copy_(grads, momentum_buffers)

        torch._foreach_add_(params, grads, alpha=-group["lr"])

    def _step_per_param(self, group) -> None:
        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            if self.maximize:
                grad = torch.neg(grad)

            state = self.state[p]

            if p.ndim > 1:
                param_norm = torch.linalg.norm(p)
                update_norm = torch.linalg.norm(grad)

                one = torch.ones_like(param_norm)
                trust_ratio = torch.where(
                    param_norm > 0.0,
                    torch.where(
                        update_norm > 0.0,
                        group["trust_coefficient"] * param_norm / update_norm,
                        one,
                    ),
                    one,
                )

                grad = grad.add(p, alpha=group["weight_decay"]).mul_(trust_ratio)

            if group["momentum"] > 0.0:
                mb = state["momentum_buffer"]
                mb.mul_(group["momentum"]).add_(grad, alpha=1.0 - group["dampening"])

                if group["nesterov"]:
                    grad = grad.add(mb, alpha=group["momentum"])
                else:
                    grad = mb

            p.add_(grad, alpha=-group["lr"])

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self._init_group(group)
            group["step"] += 1

            if self._can_use_foreach(group) and group["momentum"] > 0.0:
                params, grads, buffers = [], [], []
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    params.append(p)
                    grads.append(p.grad)
                    buffers.append(self.state[p]["momentum_buffer"])
                if params:
                    self._step_foreach(group, params, grads, buffers)
            else:
                self._step_per_param(group)

        return loss
