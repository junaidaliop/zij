# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa0)
# Copyright (c) Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# A PyTorch port of Jianlin Su's official TensorFlow Tiger from bojone/tiger.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the Tiger optimizer."""

from typing import Callable, Optional

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["Tiger"]


class Tiger(Optimizer):
    r"""Implements Tiger, a tight-fisted sign-momentum optimizer.

    Tiger keeps a single momentum buffer and forms the update from its sign,
    which makes it a SignSGD variant with momentum and weight decay, and a
    special case of Lion with :math:`\beta_1 = \beta_2 = \beta`:

    .. math::
       \begin{aligned}
       m_t &= \beta m_{t-1} + (1 - \beta)\, g_t \\
       \theta_t &= (1 - \gamma \lambda)\, \theta_{t-1}
                   - \gamma \mathrm{sign}(m_t)
       \end{aligned}

    where :math:`m_t` is the momentum buffer, :math:`\gamma` the learning rate,
    :math:`\beta` the momentum decay, and :math:`\lambda` the decoupled weight
    decay. With ``weight_decouple=False`` the decay is instead added to the
    gradient as an L2 penalty, and with ``fixed_decay=True`` the decoupled decay
    factor is :math:`(1 - \lambda)` rather than :math:`(1 - \gamma \lambda)`.

    Reference: Jianlin Su, "Tiger: A Tight-fisted Optimizer", GitHub 2023.
    https://github.com/bojone/tiger
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        beta: float = 0.965,
        weight_decay: float = 0.01,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        maximize: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.maximize = maximize

        defaults = dict(
            lr=lr,
            beta=beta,
            weight_decay=weight_decay,
            weight_decouple=weight_decouple,
            fixed_decay=fixed_decay,
        )
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
            beta = group["beta"]
            weight_decay = group["weight_decay"]
            weight_decouple = group["weight_decouple"]
            fixed_decay = group["fixed_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Tiger does not support sparse gradients.")
                if self.maximize:
                    grad = grad.neg()

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(grad)

                if weight_decouple:
                    p.mul_(1.0 - weight_decay * (1.0 if fixed_decay else lr))
                elif weight_decay > 0.0:
                    grad = grad.add(p, alpha=weight_decay)

                exp_avg = state["exp_avg"]
                exp_avg.mul_(beta).add_(grad, alpha=1.0 - beta)

                update = torch.sgn(exp_avg) if torch.is_complex(exp_avg) else torch.sign(exp_avg)
                p.add_(update, alpha=-lr)

        return loss
