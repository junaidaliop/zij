# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa02cb66)
# Copyright (c) 2021 Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the Amos optimizer."""

import math
from typing import Callable, Optional

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["Amos"]


class Amos(Optimizer):
    r"""Implements Amos, an Adam-style optimizer with adaptive weight decay
    towards a model-oriented scale.

    Amos replaces the tuned weight decay of Adam with a decay schedule driven by
    a per-variable model-oriented scale :math:`\xi`, an estimate of the magnitude
    each weight should settle at. The second moment is a scalar mean of the
    squared gradient, so the running buffers are size one per parameter tensor.

    .. math::
       \begin{aligned}
            \tilde{v}_t &= \beta\, \tilde{v}_{t-1}
                + (1 - \beta)\, \overline{g_t^2}                                 \\
            r_{v,t} &= \frac{1 - \beta^t}{\tilde{v}_t + \epsilon}                \\
            c_t &= \frac{1}{\sqrt{1 + c\,\sqrt{\eta}\; b_{t-1}}}                  \\
            d_t &= \frac{1}{1 + d\,\sqrt{\eta\,\xi}\; b_{t-1}}                    \\
            \gamma_t &= c_t\, \eta^2\, r_{v,t}\, \overline{g_t^2}                \\
            \theta_t &= \theta_{t-1} - d_t \left(
                \eta\,\xi\,\sqrt{r_{v,t}}\; g_t
                + \bigl(\tfrac{1}{2}\gamma_t + \lambda\bigr)\, \theta_{t-1} \right)  \\
            b_t &= b_{t-1}\,(1 + \gamma_t) + \gamma_t
       \end{aligned}

    where :math:`\overline{g_t^2}` is the mean of the squared gradient over the
    parameter tensor, :math:`\xi` is the model-oriented scale returned by
    :meth:`get_scale`, :math:`b_t` is the accumulated decay buffer, :math:`c` and
    :math:`d` are the decay coefficients ``c_coef`` and ``d_coef``, and
    :math:`\lambda` is the additional L2 term ``extra_l2``. An optional moving
    average of the update with rate ``momentum`` is applied before the step.

    Reference: Ran Tian, Ankur P. Parikh, "Amos: An Adam-style Optimizer with
    Adaptive Weight Decay towards Model-Oriented Scale", 2022.
    https://arxiv.org/abs/2210.11693
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        beta: float = 0.999,
        momentum: float = 0.0,
        extra_l2: float = 0.0,
        c_coef: float = 0.25,
        d_coef: float = 0.25,
        eps: float = 1e-18,
        maximize: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if extra_l2 < 0.0:
            raise ValueError(f"Invalid extra_l2 value: {extra_l2}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        self.c_coef = c_coef
        self.d_coef = d_coef
        self.maximize = maximize

        defaults = {
            "lr": lr,
            "beta": beta,
            "momentum": momentum,
            "extra_l2": extra_l2,
            "eps": eps,
        }
        super().__init__(params, defaults)

    @staticmethod
    def get_scale(p: torch.Tensor) -> float:
        r"""Get the expected model-oriented scale for the weights."""
        if len(p.shape) == 1:
            return 0.5
        if len(p.shape) == 2:
            return math.sqrt(2 / p.size(1))
        return math.sqrt(1 / p.size(1))

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

            momentum, beta = group["momentum"], group["beta"]

            lr_sq = math.sqrt(group["lr"])
            lr_p2 = math.pow(group["lr"], 2)
            bias_correction = 1.0 - math.pow(beta, group["step"])

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Amos does not support sparse gradients")

                if self.maximize:
                    grad = grad.neg()

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg_sq"] = torch.zeros(
                        (1,), dtype=p.dtype, device=p.device
                    )
                    state["decay"] = torch.zeros((1,), dtype=p.dtype, device=p.device)
                    if momentum > 0.0:
                        state["exp_avg"] = torch.zeros_like(p)

                g2 = grad.pow(2).mean()
                init_lr = group["lr"] * self.get_scale(p)

                exp_avg_sq = state["exp_avg_sq"]
                exp_avg_sq.mul_(beta).add_(g2, alpha=1.0 - beta)

                r_v_hat = bias_correction / (exp_avg_sq + group["eps"])

                decay = state["decay"]
                decay_factor_c = torch.rsqrt(1.0 + self.c_coef * lr_sq * decay)
                decay_factor_d = torch.reciprocal(
                    1.0 + self.d_coef * math.sqrt(init_lr) * decay
                )

                gamma = decay_factor_c * lr_p2 * r_v_hat * g2

                update = p.clone()
                update.mul_(0.5 * gamma + group["extra_l2"])
                update.add_(r_v_hat.sqrt() * grad, alpha=init_lr)
                update.mul_(decay_factor_d)

                decay.mul_(1.0 + gamma).add_(gamma)

                if momentum > 0.0:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(momentum).add_(update, alpha=1.0 - momentum)
                    update.copy_(exp_avg)

                p.add_(-update)

        return loss
