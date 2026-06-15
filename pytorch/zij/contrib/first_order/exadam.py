# Adapted from https://github.com/AhmedMostafa16/EXAdam (commit 9cc63be)
# Copyright (c) 2024 Ahmed M. Adly. Licensed under the MIT license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the EXAdam optimizer."""

from typing import Callable, Optional

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["EXAdam"]


class EXAdam(Optimizer):
    r"""Implements EXAdam, an extension of Adam with adaptive cross-moment
    debiasing and a gradient-based acceleration term.

    Following the official implementation, decoupled (AdamW-style) weight decay
    is applied first when ``weight_decay`` is nonzero; it is not part of the
    paper's Algorithm 1.

    .. math::
       \begin{aligned}
            \theta_{t-1} &\leftarrow \theta_{t-1} - \eta \lambda \theta_{t-1}
                \quad(\text{decoupled weight decay})                             \\
            m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t                            \\
            v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2                          \\
            d_1 &= 1 + \frac{v_t}{v_t + \epsilon} \beta_2^t                       \\
            d_2 &= 1 + \frac{m_t^2}{m_t^2 + \epsilon} \beta_1^t                   \\
            \tilde{m}_t &= \frac{m_t}{1 - \beta_1^t} \, d_1                       \\
            \tilde{v}_t &= \frac{v_t}{1 - \beta_2^t} \, d_2                       \\
            \tilde{g}_t &= \frac{g_t}{1 - \beta_1^t} \, d_1                       \\
            \theta_t &= \theta_{t-1} - \eta \,
                \frac{\tilde{m}_t + \tilde{g}_t}{\sqrt{\tilde{v}_t} + \epsilon}
       \end{aligned}

    The cross-moment factors :math:`d_1` and :math:`d_2` rescale the first and
    second moments using information from the other moment, sharpening the bias
    correction in early steps. The current gradient enters the numerator through
    :math:`\tilde{g}_t`, which accelerates the update along the instantaneous
    descent direction.

    Reference: Ahmed M. Adly, "EXAdam: The Power of Adaptive Cross-Moments",
    arXiv 2024.
    https://arxiv.org/abs/2412.20302
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
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
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("EXAdam does not support sparse gradients")

                state = self.state[p]

                if weight_decay != 0.0:
                    p.add_(p, alpha=-lr * weight_decay)

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                m, v = state["m"], state["v"]

                state["step"] += 1
                step = state["step"]

                beta1_t = beta1**step
                beta2_t = beta2**step
                bias_correction1 = 1.0 - beta1_t
                bias_correction2 = 1.0 - beta2_t

                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                d1 = 1.0 + v.div(v + eps) * beta2_t
                d2 = 1.0 + m.pow(2).div(m.pow(2) + eps) * beta1_t

                m_tilde = m.div(bias_correction1) * d1
                v_tilde = v.div(bias_correction2) * d2
                g_tilde = grad.div(bias_correction1) * d1

                p.add_(
                    (m_tilde + g_tilde) / (v_tilde.sqrt() + eps),
                    alpha=-lr,
                )

        return loss
