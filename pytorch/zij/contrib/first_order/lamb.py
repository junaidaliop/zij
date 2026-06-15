# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa0)
# Copyright (c) 2021 Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the Lamb optimizer."""

from typing import Callable, Optional

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["Lamb"]


class Lamb(Optimizer):
    r"""Implements Lamb, layer-wise adaptive optimization for large-batch training.

    Lamb rescales each layer's Adam-style update by the ratio between the norm of
    the parameters and the norm of the update (the trust ratio), so that every
    layer advances by a comparable relative amount regardless of its gradient
    scale. This follows the v3 formulation, which omits the first-moment
    de-biasing of the update and applies decoupled weight decay.

    .. math::
       \begin{aligned}
            m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t                          \\
            v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2                         \\
            r_t &= \frac{m_t}{\sqrt{v_t} + \epsilon}                            \\
            \theta_{t-1} &\leftarrow (1 - \eta \lambda)\, \theta_{t-1}          \\
            \theta_t &= \theta_{t-1} - \frac{\eta}{1 - \beta_1^t}
                \frac{\phi(\lVert \theta_{t-1} \rVert)}{\lVert r_t \rVert} r_t
       \end{aligned}

    where :math:`\lambda` is the decoupled weight decay and the trust ratio uses
    :math:`\phi(\lVert \theta \rVert) = \min(\lVert \theta \rVert, 10)`. The trust
    ratio is set to one whenever the parameter norm or the update norm is zero.

    Reference: Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar,
    Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, Cho-Jui
    Hsieh, "Large Batch Optimization for Deep Learning: Training BERT in 76
    minutes", ICLR 2020.
    https://arxiv.org/abs/1904.00962
    """

    clamp: float = 10.0

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        grad_averaging: bool = True,
        max_grad_norm: float = 1.0,
        adam: bool = False,
        pre_norm: bool = False,
        eps: float = 1e-6,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= max_grad_norm:
            raise ValueError(f"Invalid max_grad_norm value: {max_grad_norm}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        self.pre_norm = pre_norm

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "weight_decouple": weight_decouple,
            "fixed_decay": fixed_decay,
            "grad_averaging": grad_averaging,
            "max_grad_norm": max_grad_norm,
            "adam": adam,
            "eps": eps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def get_global_gradient_norm(self) -> torch.Tensor:
        device = self.param_groups[0]["params"][0].device
        global_grad_norm = torch.zeros(1, dtype=torch.float32, device=device)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    global_grad_norm.add_(p.grad.norm().pow(2))

        global_grad_norm.sqrt_().add_(self.defaults["eps"])

        return torch.clamp(
            self.defaults["max_grad_norm"] / global_grad_norm, max=1.0
        )

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        grad_norm = 1.0
        if self.pre_norm and self.defaults["max_grad_norm"] != 0.0:
            grad_norm = self.get_global_gradient_norm()

        for group in self.param_groups:
            if "step" not in group:
                group["step"] = 0
            group["step"] += 1

            beta1, beta2 = group["betas"]
            beta3 = 1.0 - beta1 if group["grad_averaging"] else 1.0
            bias_correction1 = 1.0 - beta1 ** group["step"]
            step_size = group["lr"] / bias_correction1
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lamb does not support sparse gradients")

                if self.pre_norm:
                    grad = grad.div(grad_norm)

                if (
                    group["weight_decay"] != 0.0
                    and not group["weight_decouple"]
                ):
                    grad = grad.add(p, alpha=group["weight_decay"])

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(grad, alpha=beta3)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if group["weight_decay"] != 0.0 and group["weight_decouple"]:
                    decay = group["weight_decay"] * (
                        1.0 if group["fixed_decay"] else group["lr"]
                    )
                    p.mul_(1.0 - decay)

                update = exp_avg / exp_avg_sq.sqrt().add(eps)

                weight_norm = torch.linalg.norm(p).clamp_(min=0.0, max=self.clamp)
                update_norm = torch.linalg.norm(update)
                if weight_norm == 0 or update_norm == 0:
                    trust_ratio = 1.0
                else:
                    trust_ratio = weight_norm / (update_norm + eps)

                if group["adam"]:
                    trust_ratio = 1.0

                p.add_(update, alpha=-step_size * trust_ratio)

        return loss
