# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa02cb66)
# Copyright (c) 2021 Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the AvaGrad optimizer."""

import math
from typing import Callable, Optional

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["AvaGrad"]


class AvaGrad(Optimizer):
    r"""Implements AvaGrad, an adaptive method that decouples the learning rate
    from adaptability.

    .. math::
       \begin{aligned}
            m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t                           \\
            \eta_t &= \frac{1}{\sqrt{v_{t-1}} + \epsilon}                        \\
            \gamma_t &= \frac{\sqrt{d}}{\lVert \eta_t \rVert_2}                  \\
            \theta_t &= \theta_{t-1} - \alpha \gamma_t \, \eta_t \odot m_t       \\
            v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
       \end{aligned}

    The per-parameter adaptive rate :math:`\eta_t` depends on the second moment
    from the previous step, and the global scalar :math:`\gamma_t` normalizes it
    by its root-mean-square over the :math:`d` parameters in the group. This
    normalization cancels the dependence of the update on the scale of the
    second moment, so the learning rate :math:`\alpha` and the adaptability
    :math:`\epsilon` can be tuned independently.

    Note: following the official implementation, the second moment is Adam-style
    bias-corrected before use (absent from the paper's Algorithm 2): the update
    uses :math:`\hat{v}_{t-1} = v_{t-1} / (1 - \beta_2^{t-1})` and the
    :math:`\gamma_t` normalization uses the current-step debias
    :math:`1 - \beta_2^{t}`.

    Reference: Pedro Savarese, David McAllester, Sudarshan Babu, Michael Maire,
    "Domain-Independent Dominance of Adaptive Methods", CVPR 2021.
    https://arxiv.org/abs/1912.01823
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-1,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-1,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "weight_decouple": weight_decouple,
            "fixed_decay": fixed_decay,
            "gamma": None,
            "eps": eps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if "step" not in group:
                group["step"] = 0
            group["step"] += 1

            beta1, beta2 = group["betas"]

            bias_correction1 = 1.0 - beta1 ** group["step"]
            bias_correction2_sq = math.sqrt(1.0 - beta2 ** group["step"])
            prev_bias_correction2_sq = math.sqrt(1.0 - beta2 ** (group["step"] - 1))

            step_size = group["lr"] / bias_correction1
            if group["step"] > 1:
                step_size = group["gamma"] * group["lr"] / bias_correction1

            squared_norm = 0.0
            num_params = 0.0

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AvaGrad does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                if group["weight_decay"] != 0.0:
                    if group["weight_decouple"]:
                        decay = (
                            group["weight_decay"]
                            if group["fixed_decay"]
                            else group["lr"] * group["weight_decay"]
                        )
                        p.mul_(1.0 - decay)
                    else:
                        grad = grad.add(p, alpha=group["weight_decay"])

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                sqrt_exp_avg_sq = exp_avg_sq.sqrt()

                if group["step"] > 1:
                    denom = sqrt_exp_avg_sq.div(prev_bias_correction2_sq).add_(
                        group["eps"]
                    )
                    p.addcdiv_(exp_avg, denom, value=-step_size)

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                param_wise_lr = sqrt_exp_avg_sq.div_(bias_correction2_sq).add_(
                    group["eps"]
                )
                squared_norm += param_wise_lr.norm(-2) ** -2
                num_params += param_wise_lr.numel()

            group["gamma"] = (
                0.0
                if num_params == 0.0
                else 1.0 / math.sqrt(squared_norm / num_params)
            )

        return loss
