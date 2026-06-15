# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 4abe697)
# Copyright (c) 2021 Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the NovoGrad optimizer."""

import math
from typing import Callable, Optional

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["NovoGrad"]


class NovoGrad(Optimizer):
    r"""Implements NovoGrad, an Adam variant with layer-wise adaptive moments.

    NovoGrad keeps a single scalar second moment per layer, the running average
    of the squared gradient norm, and normalizes the gradient by it before the
    first moment accumulation. Weight decay is folded into the first moment, so
    a layer with a large gradient norm receives a proportionally smaller update.

    .. math::
       \begin{aligned}
            v_t &= \beta_2 v_{t-1} + (1 - \beta_2) \lVert g_t \rVert^2          \\
            m_t &= \beta_1 m_{t-1} +
                \left( \frac{g_t}{\sqrt{v_t} + \epsilon} + \lambda \theta_{t-1}
                \right)                                                         \\
            \theta_t &= \theta_{t-1} - \eta \,
                \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t} \, m_t
       \end{aligned}

    The second moment is initialized to :math:`\lVert g_1 \rVert^2` and the
    norm is taken over each parameter tensor (the layer). With decoupled weight
    decay the :math:`\lambda \theta` term is applied directly to the parameter
    rather than through the first moment, matching AdamW. The implementation
    applies Adam-style bias correction to the step size,
    :math:`\eta \, \sqrt{1 - \beta_2^t} / (1 - \beta_1^t)`, unlike the paper
    which removes bias through initialization.

    Reference: Boris Ginsburg, Patrice Castonguay, Oleksii Hrinchuk,
    Oleksii Kuchaiev, Vitaly Lavrukhin, Ryan Leary, Jason Li, Huyen Nguyen,
    Yang Zhang, Jonathan M. Cohen, "Stochastic Gradient Methods with Layer-wise
    Adaptive Moments for Training of Deep Networks", arXiv 2019.
    https://arxiv.org/abs/1905.11286
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.95, 0.98),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        grad_averaging: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        self.maximize = maximize

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "weight_decouple": weight_decouple,
            "fixed_decay": fixed_decay,
            "grad_averaging": grad_averaging,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def _init_group(self, group: dict) -> None:
        if "step" not in group:
            group["step"] = 0

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("NovoGrad does not support sparse gradients")
            if torch.is_complex(p):
                raise RuntimeError("NovoGrad does not support complex parameters")

            state = self.state[p]
            if len(state) == 0:
                grad_p2 = grad.pow(2).sum()
                state["grads_ema"] = grad_p2
                state["moments"] = grad.div(grad_p2.sqrt().add_(group["eps"])).add_(
                    p, alpha=group["weight_decay"]
                )

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

            beta1, beta2 = group["betas"]

            bias_correction1 = 1.0 - beta1 ** group["step"]
            bias_correction2_sq = math.sqrt(1.0 - beta2 ** group["step"])
            step_size = group["lr"] * bias_correction2_sq / bias_correction1

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if self.maximize:
                    grad.neg_()

                state = self.state[p]
                grads_ema, moments = state["grads_ema"], state["moments"]

                grads_ema.mul_(beta2).add_(grad.pow(2).sum(), alpha=1.0 - beta2)

                denom = grads_ema.sqrt().add_(group["eps"])
                grad.div_(denom)

                weight_decay = group["weight_decay"]
                if weight_decay > 0.0:
                    if group["weight_decouple"]:
                        p.mul_(1.0 - weight_decay * (1.0 if group["fixed_decay"] else group["lr"]))
                    else:
                        grad.add_(p, alpha=weight_decay)

                if group["grad_averaging"]:
                    grad.mul_(1.0 - beta1)

                moments.mul_(beta1).add_(grad)

                p.add_(moments, alpha=-step_size)

        return loss
