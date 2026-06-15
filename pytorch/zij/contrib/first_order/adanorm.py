# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa0)
# Copyright (c) Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the AdaNorm optimizer."""

import math
from typing import Callable, Optional

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["AdaNorm"]


class AdaNorm(Optimizer):
    r"""Implements AdaNorm, an Adam variant with adaptive gradient norm
    correction.

    AdaNorm tracks an exponential moving average of the gradient norm and, when
    the norm of the current gradient falls below that average, rescales the
    gradient up to the running norm before it enters the first moment. This
    keeps the first moment driven by a high and representative gradient
    magnitude throughout training, while the second moment continues to use the
    raw gradient.

    .. math::
       \begin{aligned}
       s_t &= r\, s_{t-1} + (1 - r)\, \lVert g_t \rVert \\
       \tilde{g}_t &=
           \begin{cases}
               \dfrac{s_t}{\lVert g_t \rVert}\, g_t & s_t > \lVert g_t \rVert \\
               g_t & \text{otherwise}
           \end{cases} \\
       m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, \tilde{g}_t \\
       v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2 \\
       \theta_t &= \theta_{t-1} - \frac{\eta}{1 - \beta_1^t}\,
           \frac{m_t}{\sqrt{v_t / (1 - \beta_2^t)} + \epsilon}
       \end{aligned}

    where :math:`\theta` are the parameters, :math:`\eta` is the learning rate,
    :math:`g_t` is the gradient, :math:`s_t` is the running gradient norm with
    decay :math:`r`, :math:`m_t` and :math:`v_t` are the first and second
    moments, and :math:`\beta_1, \beta_2` are their decay rates.

    Reference: Shiv Ram Dubey, Satish Kumar Singh, Bidyut Baran Chaudhuri, "AdaNorm: Adaptive
    Gradient Norm Correction based Optimizer for CNNs", WACV 2023.
    https://arxiv.org/abs/2210.06364
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.99),
        r: float = 0.95,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        ams_bound: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        self.maximize = maximize

        defaults = {
            "lr": lr,
            "betas": betas,
            "r": r,
            "weight_decay": weight_decay,
            "weight_decouple": weight_decouple,
            "fixed_decay": fixed_decay,
            "ams_bound": ams_bound,
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

            bias_correction1 = 1.0 - math.pow(beta1, group["step"])
            bias_correction2_sq = math.sqrt(1.0 - math.pow(beta2, group["step"]))

            step_size = group["lr"] / bias_correction1

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdaNorm does not support sparse gradients")
                if torch.is_complex(p):
                    raise RuntimeError("AdaNorm does not support complex parameters")

                if self.maximize:
                    grad = grad.neg()

                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_var"] = torch.zeros_like(p)
                    state["exp_grad_norm"] = torch.zeros(
                        (1,), dtype=p.dtype, device=p.device
                    )
                    if group["ams_bound"]:
                        state["max_exp_avg_var"] = torch.zeros_like(p)

                exp_avg, exp_avg_var = state["exp_avg"], state["exp_avg_var"]

                weight_decay = group["weight_decay"]
                if group["weight_decouple"]:
                    p.mul_(
                        1.0
                        - weight_decay * (1.0 if group["fixed_decay"] else group["lr"])
                    )
                elif weight_decay > 0.0:
                    grad = grad.add(p, alpha=weight_decay)

                exp_grad_norm = state["exp_grad_norm"]
                grad_norm = torch.linalg.norm(grad)
                exp_grad_norm.mul_(group["r"]).add_(grad_norm, alpha=1.0 - group["r"])
                if exp_grad_norm > grad_norm:
                    s_grad = grad.mul(exp_grad_norm).div_(grad_norm)
                else:
                    s_grad = grad

                exp_avg.mul_(beta1).add_(s_grad, alpha=1.0 - beta1)
                exp_avg_var.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if group["ams_bound"]:
                    max_exp_avg_var = state["max_exp_avg_var"]
                    torch.maximum(max_exp_avg_var, exp_avg_var, out=max_exp_avg_var)
                    de_nom = max_exp_avg_var.sqrt().add_(group["eps"])
                else:
                    de_nom = exp_avg_var.sqrt().add_(group["eps"])

                de_nom.div_(bias_correction2_sq)

                p.addcdiv_(exp_avg, de_nom, value=-step_size)

        return loss
