# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa02cb66)
# Copyright (c) 2021 Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the Adai optimizer."""

import math
from typing import Callable, Optional

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["Adai"]


class Adai(Optimizer):
    r"""Implements Adai (Adaptive Inertia), which disentangles the adaptive
    learning rate of Adam into a parameter-wise adaptive momentum.

    .. math::
       \begin{aligned}
            v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2                          \\
            \hat{v}_t &= \frac{v_t}{1 - \beta_2^t}                                \\
            \beta_{1,t} &= \left( 1 - \beta_0 \frac{\hat{v}_t}{\bar{v}_t} \right)
                \text{ clamped to } [0, 1 - \epsilon]                            \\
            m_t &= \beta_{1,t} \, m_{t-1} + (1 - \beta_{1,t}) g_t                 \\
            \hat{m}_t &= \frac{m_t}{1 - \prod_{i=1}^{t} \beta_{1,i}}              \\
            \theta_t &= \theta_{t-1} - \eta \, \hat{m}_t
       \end{aligned}

    Unlike Adam, the adaptive second moment is not used to scale the step size
    directly. Instead it modulates a parameter-wise inertia (momentum) factor
    :math:`\beta_{1,t}`: parameters whose bias-corrected second moment
    :math:`\hat{v}_t` is large relative to the mean :math:`\bar{v}_t` over all
    parameters receive a smaller momentum, while parameters with small second
    moment are driven by heavier inertia. The first moment uses a per-parameter
    cumulative product of the inertia factors for bias correction.

    The ``dampening`` argument generalizes the rule: with :math:`d` the dampening,
    the inertia exponent becomes :math:`1 / (3 - 2 d)`, the gradient is scaled by
    :math:`(1 - \beta_{1,t})^d`, and the update is rescaled by
    :math:`\beta_0^{1 - d}`. The default :math:`d = 1` recovers the published
    Adai update.

    Reference: Zeke Xie, Xinrui Wang, Huishuai Zhang, Issei Sato, Masashi
    Sugiyama, "Adaptive Inertia: Disentangling the Effects of Adaptive Learning
    Rate and Momentum", ICML 2022.
    https://arxiv.org/abs/2006.15815
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.1, 0.99),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        stable_weight_decay: bool = False,
        dampening: float = 1.0,
        eps: float = 1e-3,
        maximize: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        self.maximize = maximize

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "weight_decouple": weight_decouple,
            "fixed_decay": fixed_decay,
            "stable_weight_decay": stable_weight_decay,
            "dampening": dampening,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def _init_group(self, group: dict) -> None:
        if "step" not in group:
            group["step"] = 0

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("Adai does not support sparse gradients")
            if torch.is_complex(p):
                raise RuntimeError("Adai does not support complex parameters")

            state = self.state[p]
            if len(state) == 0:
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
                state["beta1_prod"] = torch.ones_like(p)

    def _apply_weight_decay(self, p: torch.Tensor, grad: torch.Tensor, group: dict) -> None:
        if group["weight_decay"] == 0.0:
            return
        if group["weight_decouple"]:
            decay = group["weight_decay"] * (
                1.0 if group["fixed_decay"] else group["lr"]
            )
            p.mul_(1.0 - decay)
        else:
            grad.add_(p, alpha=group["weight_decay"])

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        param_size = 0
        exp_avg_sq_hat_sum = 0.0

        for group in self.param_groups:
            self._init_group(group)
            group["step"] += 1

            _, beta2 = group["betas"]
            bias_correction2 = 1.0 - beta2 ** group["step"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if self.maximize:
                    grad = grad.neg()

                param_size += p.numel()

                state = self.state[p]

                if not group["stable_weight_decay"]:
                    self._apply_weight_decay(p, grad, group)

                exp_avg_sq = state["exp_avg_sq"]
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                exp_avg_sq_hat_sum += exp_avg_sq.sum() / bias_correction2

        if param_size == 0:
            raise ValueError("Adai got an empty parameter list")

        exp_avg_sq_hat_mean = exp_avg_sq_hat_sum / param_size

        for group in self.param_groups:
            beta0, beta2 = group["betas"]

            beta0_dp = math.pow(beta0, 1.0 - group["dampening"])
            bias_correction2 = 1.0 - beta2 ** group["step"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if self.maximize:
                    grad = grad.neg()

                state = self.state[p]

                if group["stable_weight_decay"]:
                    self._apply_weight_decay(p, grad, group)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                exp_avg_sq_hat = exp_avg_sq / bias_correction2
                beta1 = (
                    1.0
                    - (exp_avg_sq_hat / exp_avg_sq_hat_mean)
                    .pow_(1.0 / (3.0 - 2.0 * group["dampening"]))
                    .mul_(beta0)
                ).clamp_(0.0, 1.0 - group["eps"])
                beta3 = (1.0 - beta1).pow_(group["dampening"])

                beta1_prod = state["beta1_prod"]
                beta1_prod.mul_(beta1)

                exp_avg.mul_(beta1).addcmul_(beta3, grad)
                exp_avg_hat = exp_avg.div(1.0 - beta1_prod).mul_(beta0_dp)

                p.add_(exp_avg_hat, alpha=-group["lr"])

        return loss
