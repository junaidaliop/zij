# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa0)
# Copyright (c) Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# Deviation from upstream: the global gradient norm is the true L2 norm
# (sqrt of the sum of squared per-tensor norms), matching the paper's GlobalGC,
# whereas upstream returns the unrooted sum of squares.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the AdaGC optimizer."""

import math
from typing import Callable, Optional

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["AdaGC"]


class AdaGC(Optimizer):
    r"""Implements AdaGC, Adam with adaptive per-tensor gradient clipping.

    AdaGC stabilizes large language model pretraining by clipping each tensor's
    gradient against an exponential moving average of its own past clipped
    gradient norms. During an initial warmup the clipping is global and the
    threshold :math:`\gamma` tracks the running minimum of the clipped norms;
    afterwards each tensor is clipped locally relative to its own history. The
    clipped gradient then drives a standard Adam update.

    .. math::
       \begin{aligned}
            h_t &= \min\!\left(
                \frac{\lambda_{\text{rel}} \, \gamma_{t-1}}{\lVert g_t \rVert},
                1 \right)                                                        \\
            \hat{g}_t &= h_t \, g_t                                              \\
            \gamma_t &= \beta \gamma_{t-1}
                + (1 - \beta) \lVert \hat{g}_t \rVert                           \\
            m_t &= \beta_1 m_{t-1} + (1 - \beta_1) \hat{g}_t                     \\
            v_t &= \beta_2 v_{t-1} + (1 - \beta_2) \hat{g}_t^2                   \\
            \theta_t &= \theta_{t-1} - \eta \,
                \frac{m_t / (1 - \beta_1^t)}
                     {\sqrt{v_t / (1 - \beta_2^t)} + \epsilon}
       \end{aligned}

    where :math:`g_t` is the gradient, :math:`\hat{g}_t` the clipped gradient,
    :math:`\gamma_t` the per-tensor exponential moving average of clipped norms,
    :math:`\lambda_{\text{rel}}` the relative clipping threshold, :math:`\beta`
    the smoothing coefficient, and :math:`m_t`, :math:`v_t` the Adam moments.
    During the first ``warmup_steps`` iterations the clipping factor uses the
    absolute threshold, :math:`h_t = \min(\lambda_{\text{abs}} / \lVert g_t
    \rVert, 1)`, and :math:`\gamma_t = \min(\gamma_{t-1}, \lVert \hat{g}_t
    \rVert)`.

    Reference: Guoxia Wang, Shuai Li, Congliang Chen, Jinle Zeng, Jiabin Yang,
    Dianhai Yu, Yanjun Ma, Li Shen, "AdaGC: Enhancing LLM Pretraining Stability
    via Adaptive Gradient Clipping", ICML 2026.
    https://arxiv.org/abs/2502.11034
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        beta: float = 0.98,
        lambda_abs: float = 1.0,
        lambda_rel: float = 1.05,
        warmup_steps: int = 100,
        weight_decay: float = 1e-1,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if not 0.0 < lambda_abs:
            raise ValueError(f"Invalid lambda_abs value: {lambda_abs}")
        if not 0.0 < lambda_rel:
            raise ValueError(f"Invalid lambda_rel value: {lambda_rel}")
        if not 0 <= warmup_steps:
            raise ValueError(f"Invalid warmup_steps value: {warmup_steps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        self.maximize = maximize

        defaults = {
            "lr": lr,
            "betas": betas,
            "beta": beta,
            "lambda_abs": lambda_abs,
            "lambda_rel": lambda_rel,
            "warmup_steps": warmup_steps,
            "weight_decay": weight_decay,
            "weight_decouple": weight_decouple,
            "fixed_decay": fixed_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("weight_decouple", True)
            group.setdefault("fixed_decay", False)

    def _global_gradient_norm(self) -> torch.Tensor:
        device = self.param_groups[0]["params"][0].device
        total = torch.zeros(1, dtype=torch.float32, device=device)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    total.add_(p.grad.norm().pow(2))
        return total.sqrt()

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
            beta = group["beta"]
            lambda_abs = group["lambda_abs"]
            lambda_rel = group["lambda_rel"]
            warmup_steps = group["warmup_steps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            weight_decouple = group["weight_decouple"]
            fixed_decay = group["fixed_decay"]
            eps = group["eps"]

            bias_correction1 = 1.0 - math.pow(beta1, group["step"])
            bias_correction2_sq = math.sqrt(1.0 - math.pow(beta2, group["step"]))

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdaGC does not support sparse gradients")
                if torch.is_complex(p):
                    raise RuntimeError("AdaGC does not support complex parameters")
                if self.maximize:
                    grad = grad.neg()

                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    state["gamma"] = torch.empty((1,), device=grad.device, dtype=grad.dtype)

                if weight_decay != 0.0:
                    if weight_decouple:
                        p.mul_(1.0 - weight_decay * (1.0 if fixed_decay else lr))
                    else:
                        grad = grad.add(p, alpha=weight_decay)

                exp_avg, exp_avg_sq, gamma = (
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["gamma"],
                )

                if group["step"] < warmup_steps:
                    grad_norm = self._global_gradient_norm().add_(eps)

                    h_t = min(lambda_abs / grad_norm, 1.0)
                    g_hat = grad.mul(h_t)

                    g_hat_norm = g_hat.norm()

                    gamma.copy_(g_hat_norm if group["step"] == 1 else min(gamma, g_hat_norm))
                else:
                    h_t = min(lambda_rel * gamma / grad.norm(), 1.0)
                    g_hat = grad.mul(h_t)

                    gamma.mul_(beta).add_(g_hat.norm(), alpha=1.0 - beta)

                exp_avg.mul_(beta1).add_(g_hat, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_hat, g_hat, value=1.0 - beta2)

                update = (exp_avg / bias_correction1) / exp_avg_sq.sqrt().div_(
                    bias_correction2_sq
                ).add_(eps)

                p.add_(update, alpha=-lr)

        return loss
