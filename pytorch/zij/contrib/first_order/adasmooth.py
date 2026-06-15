# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa0)
# Copyright (c) Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the AdaSmooth optimizer."""

from typing import Callable, Optional

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["AdaSmooth"]


class AdaSmooth(Optimizer):
    r"""Implements AdaSmooth, an adaptive learning rate method based on the
    effective ratio.

    AdaSmooth replaces the fixed decay of the squared-gradient running average
    with a per-parameter smoothing constant derived from the effective ratio of
    the recent parameter trajectory. The effective ratio measures how directed
    the movement has been: it is the magnitude of the accumulated change divided
    by the accumulated absolute change. A directed trajectory (ratio near one)
    yields a short averaging window, which speeds up the descent, while a
    zigzagging trajectory (ratio near zero) yields a long window, which slows
    the descent near a minimum.

    .. math::
       \begin{aligned}
            s_t &= s_{t-1} + (\theta_t - \theta_{t-1})                          \\
            n_t &= n_{t-1} + |\theta_t - \theta_{t-1}|                          \\
            e_t &= \frac{\left| \sum s_t \right|}{\sum n_t}                     \\
            c_t &= (\rho_2 - \rho_1)\, e_t + (1 - \rho_2)                       \\
            v_t &= c_t^2\, g_t^2 + (1 - c_t^2)\, v_{t-1}                        \\
            \theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{v_t + \epsilon}}\, g_t
       \end{aligned}

    where :math:`\theta` are the parameters, :math:`g_t` is the gradient,
    :math:`s_t` and :math:`n_t` accumulate the signed and absolute parameter
    changes, :math:`e_t` is the effective ratio, :math:`c_t` is the scaled
    smoothing constant built from the fast and slow decay rates
    :math:`\rho_1, \rho_2` (passed as ``betas``), :math:`v_t` is the running
    average of the squared gradient, :math:`\eta` is the learning rate, and
    :math:`\epsilon` guards the denominator.

    Reference: Jun Lu, "AdaSmooth: An Adaptive Learning Rate Method based on
    Effective Ratio", arXiv 2022.
    https://arxiv.org/abs/2204.00825
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.5, 0.99),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        eps: float = 1e-6,
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
            "weight_decay": weight_decay,
            "weight_decouple": weight_decouple,
            "fixed_decay": fixed_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("weight_decouple", False)
            group.setdefault("fixed_decay", False)

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
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            weight_decouple = group["weight_decouple"]
            fixed_decay = group["fixed_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdaSmooth does not support sparse gradients")
                if self.maximize:
                    grad = grad.neg()

                state = self.state[p]

                if len(state) == 0:
                    state["prev_param"] = torch.zeros_like(p)
                    state["s"] = torch.zeros_like(p)
                    state["n"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                s, n = state["s"], state["n"]
                prev_param, exp_avg_sq = state["prev_param"], state["exp_avg_sq"]

                if torch.is_complex(p):
                    p = torch.view_as_real(p)
                    grad = torch.view_as_real(grad)
                    s = torch.view_as_real(s)
                    n = torch.view_as_real(n)
                    prev_param = torch.view_as_real(prev_param)
                    exp_avg_sq = torch.view_as_real(exp_avg_sq)

                if weight_decay != 0.0:
                    if weight_decouple:
                        p.mul_(1.0 - weight_decay * (1.0 if fixed_decay else lr))
                    else:
                        grad = grad.add(p, alpha=weight_decay)

                p_diff = p - prev_param
                s.add_(p_diff)
                n.add_(p_diff.abs())

                c = s.sum().abs_().div_(n.sum())
                c.mul_(beta2 - beta1).add_(1.0 - beta2)

                c_p2 = c.pow(2)
                exp_avg_sq.mul_(1.0 - c_p2).addcmul_(grad, grad, value=c_p2)

                step_size = torch.full_like(exp_avg_sq, fill_value=lr)
                step_size.div_((exp_avg_sq + eps).sqrt()).mul_(grad)

                p.add_(-step_size)

                prev_param.copy_(p)

        return loss
