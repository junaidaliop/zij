# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa0)
# Copyright (c) Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the Ranger optimizer."""

import math
from typing import Callable, Optional

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["Ranger"]


def _centralize_gradient(grad: torch.Tensor, gc_conv_only: bool = False) -> None:
    """Subtract the per-tensor gradient mean (Gradient Centralization)."""
    size = grad.dim()
    if (gc_conv_only and size > 3) or (not gc_conv_only and size > 1):
        grad.add_(-grad.mean(dim=tuple(range(1, size)), keepdim=True))


class Ranger(Optimizer):
    r"""Implements Ranger, RAdam with a Lookahead wrapper and gradient
    centralization.

    Each step takes a rectified Adam (RAdam) update on the fast weights and,
    every :math:`k` steps, interpolates a set of slow weights toward them
    (Lookahead). Gradients are optionally centralized by subtracting their mean
    before the moment updates.

    .. math::
       \begin{aligned}
       m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
       v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
       \rho_\infty &= \frac{2}{1 - \beta_2} - 1 \\
       \rho_t &= \rho_\infty - \frac{2 t\, \beta_2^t}{1 - \beta_2^t}
       \end{aligned}

    When the length of the approximated simple moving average satisfies
    :math:`\rho_t \geq` ``n_sma_threshold`` (default 5), the variance is
    tractable and the step is rectified:

    .. math::
       \begin{aligned}
       r_t &= \sqrt{\frac{(1 - \beta_2^t)(\rho_t - 4)(\rho_t - 2)\rho_\infty}
                         {(\rho_\infty - 4)(\rho_\infty - 2)\rho_t}} \\
       \theta_t &= \theta_{t-1} - \frac{\eta\, r_t}{1 - \beta_1^t}
                    \frac{m_t}{\sqrt{v_t} + \epsilon}
       \end{aligned}

    Otherwise, with ``degenerated_to_sgd=True``, the update falls back to the
    unscaled first moment,
    :math:`\theta_t = \theta_{t-1} - \frac{\eta}{1 - \beta_1^t} m_t`; with the
    default ``degenerated_to_sgd=False`` the rectified branch is simply skipped
    until the moving average becomes tractable. Every
    :math:`k` steps the slow weights :math:`\phi` track the fast weights,
    :math:`\phi_t = \phi_{t-1} + \alpha (\theta_t - \phi_{t-1})`, and the fast
    weights are reset to :math:`\phi_t`.

    Here :math:`\theta` are the parameters, :math:`\eta` is the learning rate,
    :math:`g_t` is the gradient, :math:`m_t` and :math:`v_t` are the first and
    second moments, :math:`\beta_1, \beta_2` are their decay rates, :math:`k` is
    the Lookahead synchronization period, and :math:`\alpha` is the Lookahead
    interpolation factor.

    Reference: Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong
    Liu, Jianfeng Gao, Jiawei Han, "On the Variance of the Adaptive Learning
    Rate and Beyond", ICLR 2020. https://arxiv.org/abs/1908.03265
    Reference: Michael R. Zhang, James Lucas, Geoffrey Hinton, Jimmy Ba,
    "Lookahead Optimizer: k steps forward, 1 step back", NeurIPS 2019.
    https://arxiv.org/abs/1907.08610
    Reference: Hongwei Yong, Jianqiang Huang, Xiansheng Hua, Lei Zhang,
    "Gradient Centralization: A New Optimization Technique for Deep Neural
    Networks", ECCV 2020. https://arxiv.org/abs/2004.01461
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.95, 0.999),
        alpha: float = 0.5,
        k: int = 6,
        n_sma_threshold: int = 5,
        degenerated_to_sgd: bool = False,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        use_gc: bool = True,
        gc_conv_only: bool = False,
        eps: float = 1e-5,
        maximize: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha: {alpha} - should be in [0.0, 1.0]")
        if k <= 0:
            raise ValueError(f"Invalid k: {k} - should be positive")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay} - should be >= 0.0")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")

        self.n_sma_threshold = n_sma_threshold
        self.degenerated_to_sgd = degenerated_to_sgd
        self.use_gc = use_gc
        self.gc_gradient_threshold = 3 if gc_conv_only else 1
        self.maximize = maximize

        defaults = {
            "lr": lr,
            "betas": betas,
            "alpha": alpha,
            "k": k,
            "weight_decay": weight_decay,
            "weight_decouple": weight_decouple,
            "fixed_decay": fixed_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def _init_group(self, group: dict) -> None:
        if "step" not in group:
            group["step"] = 0

        for p in group["params"]:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError("Ranger does not support sparse gradients")

            state = self.state[p]
            if len(state) == 0:
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
                state["slow_buffer"] = p.clone()

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
            step = group["step"]

            beta1, beta2 = group["betas"]
            bias_correction1 = 1.0 - beta1**step

            n_sma_max = 2.0 / (1.0 - beta2) - 1.0
            beta2_t = beta2**step
            n_sma = n_sma_max - 2 * step * beta2_t / (1.0 - beta2_t)

            if n_sma >= self.n_sma_threshold:
                rt = math.sqrt(
                    (1.0 - beta2_t)
                    * (n_sma - 4)
                    / (n_sma_max - 4)
                    * (n_sma - 2)
                    / n_sma
                    * n_sma_max
                    / (n_sma_max - 2)
                )
            elif self.degenerated_to_sgd:
                rt = 1.0
            else:
                rt = -1.0

            step_size = group["lr"] * rt / bias_correction1

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if self.maximize:
                    grad.neg_()

                state = self.state[p]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                slow_buffer = state["slow_buffer"]

                if self.use_gc and grad.dim() > self.gc_gradient_threshold:
                    _centralize_gradient(grad, gc_conv_only=False)

                if group["weight_decouple"]:
                    p.mul_(
                        1.0
                        - group["weight_decay"]
                        * (1.0 if group["fixed_decay"] else group["lr"])
                    )
                elif group["weight_decay"] > 0.0:
                    grad.add_(p, alpha=group["weight_decay"])

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if n_sma >= self.n_sma_threshold:
                    de_nom = exp_avg_sq.sqrt().add_(group["eps"])
                    p.addcdiv_(exp_avg, de_nom, value=-step_size)
                elif step_size > 0:
                    # When the variance is not yet tractable, only take the
                    # momentum-only/SGD step if degenerated_to_sgd produced a
                    # positive step size; the rt=-1 sentinel skips the step
                    # rather than ascending (matches the RAdam paper).
                    p.add_(exp_avg, alpha=-step_size)

                if step % group["k"] == 0:
                    slow_buffer.lerp_(p, weight=group["alpha"])
                    p.copy_(slow_buffer)

        return loss
