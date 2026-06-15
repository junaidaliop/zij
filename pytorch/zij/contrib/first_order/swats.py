# Adapted from https://github.com/jettify/pytorch-optimizer (commit 19c3e41)
# Copyright (c) Nikolay Novik (jettify). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the SWATS optimizer."""

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["SWATS"]


class SWATS(Optimizer):
    r"""Implements SWATS, switching from Adam to SGD during training.

    Each parameter group starts in an Adam phase. After every Adam step the
    method estimates the learning rate an equivalent SGD update would use by
    projecting the Adam step :math:`p_t` onto the gradient, and tracks a
    bias-corrected running average :math:`\Lambda_t` of that estimate. When the
    estimate stabilizes, the group switches to SGD with momentum using
    :math:`\Lambda_t` as its learning rate.

    .. math::
       \begin{aligned}
            m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t                          \\
            v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2                         \\
            p_t &= -\frac{\eta\,\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}
                \frac{m_t}{\sqrt{v_t} + \epsilon}                               \\
            \gamma_t &= \frac{p_t^\top p_t}{-\,p_t^\top g_t}                     \\
            \lambda_t &= \beta_2 \lambda_{t-1} + (1 - \beta_2)\gamma_t,
                \qquad \Lambda_t = \lambda_t / (1 - \beta_2^t)
       \end{aligned}

    When :math:`\Lambda_t \approx \gamma_t` and :math:`\Lambda_t > 0`, the group
    switches to SGD with learning rate :math:`\Lambda_t`.

    Reference: Nitish Shirish Keskar, Richard Socher, "Improving Generalization
    Performance by Switching from Adam to SGD", 2017.
    https://arxiv.org/abs/1712.07628
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-3,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        nesterov: bool = False,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "phase": "ADAM",
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
            "nesterov": nesterov,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for w in group["params"]:
                if w.grad is None:
                    continue
                grad = w.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "SWATS does not support sparse gradients, "
                        "please consider SparseAdam instead"
                    )

                amsgrad = group["amsgrad"]
                state = self.state[w]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        w, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        w, memory_format=torch.preserve_format
                    )
                    # moving average for the non-orthogonal projection scaling
                    state["exp_avg2"] = w.new_zeros(1)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            w, memory_format=torch.preserve_format
                        )

                exp_avg, exp_avg2, exp_avg_sq = (
                    state["exp_avg"],
                    state["exp_avg2"],
                    state["exp_avg_sq"],
                )
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(w, alpha=group["weight_decay"])

                # if it is the SGD phase, take an SGD update and continue
                if group["phase"] == "SGD":
                    if "momentum_buffer" not in state:
                        buf = state["momentum_buffer"] = torch.clone(
                            grad
                        ).detach()
                    else:
                        buf = state["momentum_buffer"]
                        buf.mul_(beta1).add_(grad)
                        grad = buf

                    grad = grad.mul(1 - beta1)
                    if group["nesterov"]:
                        grad = grad.add(buf, alpha=beta1)

                    w.add_(grad, alpha=-group["lr"])
                    continue

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = (
                    group["lr"] * (bias_correction2**0.5) / bias_correction1
                )

                p = -step_size * (exp_avg / denom)
                w.add_(p)

                p_view = p.view(-1)
                pg = p_view.dot(grad.view(-1))

                if pg != 0:
                    # the non-orthogonal scaling estimate
                    scaling = p_view.dot(p_view) / -pg
                    exp_avg2.mul_(beta2).add_(scaling, alpha=1 - beta2)

                    # bias-corrected exponential average
                    corrected_exp_avg = exp_avg2 / bias_correction2

                    # check the criterion for switching to SGD training
                    if (
                        state["step"] > 1
                        and corrected_exp_avg.allclose(scaling, rtol=1e-6)
                        and corrected_exp_avg > 0
                    ):
                        group["phase"] = "SGD"
                        group["lr"] = corrected_exp_avg.item()
        return loss
