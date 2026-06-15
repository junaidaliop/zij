# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa0)
# Copyright (c) Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the SGD-SaI optimizer."""

from typing import Callable, Optional

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["SGDSaI"]


class SGDSaI(Optimizer):
    r"""Implements SGD-SaI, SGD with momentum and learning rate Scaling at Initialization.

    SGD-SaI replaces Adam's second-order momentum with a per-block scaling
    factor, the gradient signal-to-noise ratio (g-SNR), computed once from the
    gradients of the first batch and then held constant for the rest of
    training. For a parameter block :math:`i` with gradient :math:`g`, the g-SNR
    is the ratio of the gradient norm to its standard deviation:

    .. math::
       G^{(i)}_{\mathrm{snr}} = \frac{\lVert g \rVert_2}{\sigma(g) + \epsilon}

    where :math:`\sigma(g)` is the standard deviation of the gradient entries.
    The block is then updated with momentum and decoupled weight decay, scaling
    the learning rate by the constant g-SNR:

    .. math::
       \begin{aligned}
       m_t &= \mu m_{t-1} + (1 - \mu)\, g_t \\
       \theta_t &= (1 - \gamma \lambda)\, \theta_{t-1}
                   - \gamma\, G^{(i)}_{\mathrm{snr}}\, m_t
       \end{aligned}

    where :math:`\gamma` is the learning rate, :math:`\mu` the momentum
    coefficient, and :math:`\lambda` the weight decay. With
    ``weight_decouple=False`` the weight decay is instead added to the gradient
    as an L2 penalty.

    Reference: Minghao Xu et al., "No More Adam: Learning Rate Scaling at
    Initialization is All You Need", 2024.
    https://arxiv.org/abs/2412.11768
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        eps: float = 1e-8,
        maximize: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        self.has_warmup: bool = False
        self.maximize = maximize

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            weight_decouple=weight_decouple,
            eps=eps,
        )
        super().__init__(params, defaults)

    def _init_group(self, group) -> None:
        if "step" not in group:
            group["step"] = 0

        for p in group["params"]:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError("SGD-SaI does not support sparse gradients.")

            state = self.state[p]
            if group["momentum"] > 0.0:
                state["momentum_buffer"] = torch.zeros_like(p)

    @torch.no_grad()
    def warmup_step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Computes and caches the per-block g-SNR from the first batch."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self._init_group(group)
            group["step"] += 1

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if self.maximize:
                    grad.neg_()

                sigma = grad.std().nan_to_num_() if grad.ndim > 1 and grad.size(0) != 1 else 0
                grad_norm = grad.norm()

                g_snr = grad_norm.div_(sigma.add_(group["eps"])) if sigma != 0.0 else grad_norm

                self.state[p]["gsnr"] = g_snr

        self.has_warmup = True

        return loss

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step."""
        if not self.has_warmup:
            self.warmup_step(closure)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["step"] += 1

            momentum = group["momentum"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            weight_decouple = group["weight_decouple"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if self.maximize:
                    grad.neg_()

                state = self.state[p]

                if momentum > 0.0:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)
                else:
                    buf = grad

                if weight_decouple:
                    p.mul_(1.0 - weight_decay * lr)
                elif weight_decay > 0.0:
                    grad.add_(p, alpha=weight_decay)

                p.add_(buf, alpha=-lr * state["gsnr"])

        return loss
