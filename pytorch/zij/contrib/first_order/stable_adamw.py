# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa0)
# Copyright (c) Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the StableAdamW optimizer."""

import math
from typing import Callable, Optional

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["StableAdamW"]


class StableAdamW(Optimizer):
    r"""Implements StableAdamW, AdamW with Adafactor-style update clipping.

    StableAdamW rescales each AdamW update by the root-mean-square of the
    per-coordinate ratio between the gradient and the second-moment estimate.
    When that ratio is large the effective learning rate is shrunk, which
    removes the loss spikes that gradient clipping leaves behind during
    low-precision training.

    .. math::
       \begin{aligned}
            \mathrm{debias\_beta}(\beta, t) &=
                \frac{\beta^t - \beta}{\beta^t - 1}                              \\
            c_1(t) &= 1 - \mathrm{debias\_beta}(\beta_1, t)                      \\
            c_2(t) &= \mathrm{debias\_beta}(\beta_2, t)                          \\
            m_t &= (1 - c_1(t)) \, m_{t-1} + c_1(t) \, g_t                       \\
            v_t &= c_2(t) \, v_{t-1} + (1 - c_2(t)) \, g_t^2                      \\
            \mathrm{RMS}_t &= \sqrt{\,
                \mathrm{mean}\!\left(
                    \frac{g_t^2}{\max(v_t, \epsilon^2)}
                \right)}                                                         \\
            \eta_t &= \frac{\eta}{\max(1, \mathrm{RMS}_t)}                       \\
            \theta_t &= \theta_{t-1} - \eta_t \,
                \frac{m_t}{\sqrt{v_t} + \epsilon}
       \end{aligned}

    Bias correction is applied through the step-dependent coefficients
    :math:`c_1(t)` and :math:`c_2(t)`, which are computed from
    :math:`\mathrm{debias\_beta}(\beta, t) = (\beta^t - \beta) / (\beta^t - 1)`
    rather than via a separate :math:`(1 - \beta^t)` normalization. The moments
    are therefore updated and consumed in their interpolated (lerp) form.
    Weight decay is decoupled by default. Optional Kahan summation compensates
    for rounding when the parameters are stored in ``float16`` or ``bfloat16``.

    Reference: Mitchell Wortsman, Tim Dettmers, Luke Zettlemoyer, Ari Morcos,
    Ali Farhadi, Ludwig Schmidt, "Stable and low-precision training for
    large-scale vision-language models", NeurIPS 2023.
    https://arxiv.org/abs/2304.13013
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        kahan_sum: bool = True,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "weight_decouple": weight_decouple,
            "kahan_sum": kahan_sum,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("weight_decouple", True)
            group.setdefault("kahan_sum", True)

    @staticmethod
    def _debias_beta(beta: float, step: int) -> float:
        beta_n = math.pow(beta, step)
        return (beta_n - beta) / (beta_n - 1.0)

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
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            weight_decouple = group["weight_decouple"]
            kahan_sum = group["kahan_sum"]

            beta1_comp = 1.0 - self._debias_beta(beta1, group["step"])
            beta2_hat = self._debias_beta(beta2, group["step"])
            eps_sq = eps * eps

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("StableAdamW does not support sparse gradients")

                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["kahan_comp"] = (
                        torch.zeros_like(p)
                        if (kahan_sum and p.dtype in (torch.float16, torch.bfloat16))
                        else None
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                exp_avg.lerp_(grad, weight=beta1_comp)
                exp_avg_sq.mul_(beta2_hat).addcmul_(grad, grad, value=1.0 - beta2_hat)

                rms = (
                    grad.pow(2)
                    .div_(exp_avg_sq.clip(min=eps_sq))
                    .mean()
                    .sqrt_()
                    .clip_(min=1.0)
                    .item()
                )
                lr = group["lr"] / rms

                if weight_decay != 0.0:
                    if weight_decouple:
                        p.mul_(1.0 - weight_decay * lr)
                    else:
                        grad = grad.add(p, alpha=weight_decay)

                denom = exp_avg_sq.sqrt().add_(eps)

                if state["kahan_comp"] is not None:
                    kahan_comp = state["kahan_comp"]
                    kahan_comp.addcdiv_(exp_avg, denom, value=-lr)

                    prev = p.detach().clone()
                    p.add_(kahan_comp)
                    kahan_comp.add_(prev.sub_(p))
                else:
                    p.addcdiv_(exp_avg, denom, value=-lr)

        return loss
