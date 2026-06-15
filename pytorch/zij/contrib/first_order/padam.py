# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa02cb66)
# Copyright (c) 2021 Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the PAdam optimizer."""

import math
from typing import Callable, Optional

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["PAdam"]


class PAdam(Optimizer):
    r"""Implements PAdam, partially adaptive momentum estimation.

    PAdam interpolates between Adam and SGD with momentum by raising the
    second-moment denominator to a partial power :math:`p \in (0, 1/2]`. With
    :math:`p = 1/2` the update is Adam; as :math:`p \to 0` the adaptivity
    vanishes and the update approaches plain momentum, which lets PAdam use a
    larger base learning rate without the gradient explosion that small
    denominators cause.

    .. math::
       \begin{aligned}
            m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t                   \\
            v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2                 \\
            \theta_t &= \theta_{t-1} - \eta \, \frac{m_t}{v_t^{\,p}}
       \end{aligned}

    This implementation applies Adam-style bias correction to the moments and
    raises the bias-corrected denominator to the partial power, so the effective
    step is :math:`\eta \, \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)^{2p}`, which
    equals :math:`\hat{v}_t^{\,p}` up to the stabilizing :math:`\epsilon`.

    Reference: Jinghui Chen, Dongruo Zhou, Yiqi Tang, Ziyan Yang, Yuan Cao,
    Quanquan Gu, "Closing the Generalization Gap of Adaptive Gradient Methods in
    Training Deep Neural Networks", IJCAI 2020.
    https://arxiv.org/abs/1806.06763
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-1,
        betas: tuple[float, float] = (0.9, 0.999),
        partial: float = 0.25,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 < partial <= 0.5:
            raise ValueError(f"Invalid partial value: {partial}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        self.maximize = maximize

        defaults = {
            "lr": lr,
            "betas": betas,
            "partial": partial,
            "weight_decay": weight_decay,
            "weight_decouple": weight_decouple,
            "fixed_decay": fixed_decay,
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

            step_size = group["lr"] * bias_correction2_sq / bias_correction1

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("PAdam does not support sparse gradients")

                if self.maximize:
                    grad = grad.neg()

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                if torch.is_complex(p):
                    p = torch.view_as_real(p)
                    grad = torch.view_as_real(grad)
                    exp_avg = torch.view_as_real(exp_avg)
                    exp_avg_sq = torch.view_as_real(exp_avg_sq)

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
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                p.addcdiv_(exp_avg, denom ** (group["partial"] * 2), value=-step_size)

        return loss
