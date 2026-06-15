# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa0)
# Copyright (c) Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the AdamG optimizer."""

import math

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["AdamG"]


class AdamG(Optimizer):
    r"""Implements AdamG, a parameter-free Adam with the golden step size.

    .. math::
       \begin{aligned}
            v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2                          \\
            r_t &= \beta_3 r_{t-1} + (1 - \beta_3) s(v_t),
                \quad s(x) = p \, x^q                                             \\
            m_t &= \beta_1 m_{t-1} + (1 - \beta_1) \, r_t \odot g_t               \\
            \hat{m}_t &= m_t / (1 - \beta_1^t), \quad
                \hat{v}_t = v_t / (1 - \beta_2^t)                                 \\
            \theta_t &= \theta_{t-1} - \min(\eta, 1/\sqrt{t}) \,
                \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)
       \end{aligned}

    Reference: Yijiang Pang, Shuyang Yu, Bao Hoang, Jiayu Zhou,
    "Towards Stability of Parameter-free Optimization", arXiv 2024.
    https://arxiv.org/abs/2405.04376
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1.0,
        betas: tuple[float, float, float] = (0.95, 0.999, 0.95),
        p: float = 0.2,
        q: float = 0.24,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        for i, beta in enumerate(betas):
            if not 0.0 <= beta < 1.0:
                raise ValueError(f"Invalid beta parameter at index {i}: {beta}")
        if not 0.0 < p:
            raise ValueError(f"Invalid p value: {p}")
        if not 0.0 < q:
            raise ValueError(f"Invalid q value: {q}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "p": p,
            "q": q,
            "weight_decay": weight_decay,
            "weight_decouple": weight_decouple,
            "fixed_decay": fixed_decay,
            "eps": eps,
            "maximize": maximize,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if "step" not in group:
                group["step"] = 0
            group["step"] += 1

            beta1, beta2, beta3 = group["betas"]

            bias_correction1 = 1.0 - beta1 ** group["step"]
            bias_correction2 = 1.0 - beta2 ** group["step"]

            step_size = min(group["lr"], 1.0 / math.sqrt(group["step"]))

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamG does not support sparse gradients")
                if group["maximize"]:
                    grad = -grad

                state = self.state[param]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(param)
                    state["v"] = torch.zeros_like(param)
                    state["r"] = torch.zeros_like(param)

                m, v, r = state["m"], state["v"], state["r"]

                if torch.is_complex(param):
                    param = torch.view_as_real(param)
                    grad = torch.view_as_real(grad)
                    m = torch.view_as_real(m)
                    v = torch.view_as_real(v)
                    r = torch.view_as_real(r)

                if group["weight_decouple"]:
                    param.mul_(
                        1.0
                        - group["weight_decay"]
                        * (1.0 if group["fixed_decay"] else group["lr"])
                    )
                elif group["weight_decay"] != 0.0:
                    grad = grad.add(param, alpha=group["weight_decay"])

                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                r.mul_(beta3).add_(
                    v.pow(group["q"]).mul_(group["p"]), alpha=1.0 - beta3
                )
                m.mul_(beta1).addcmul_(r, grad, value=1.0 - beta1)

                update = (m / bias_correction1).div_(
                    (v / bias_correction2).sqrt_().add_(group["eps"])
                )

                param.add_(update, alpha=-step_size)

        return loss
