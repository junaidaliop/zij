# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa0)
# Copyright (c) Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the SPlus optimizer."""

from typing import Callable, Optional, Tuple

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["SPlus"]


class SPlus(Optimizer):
    r"""Implements SPlus, a stable whitening optimizer.

    SPlus preconditions each matrix parameter with a Kronecker-factored,
    Shampoo-style whitening of the gradient, but replaces the cached
    matrix-inverse update with a bounded one that pairs a slowly updated
    eigenbasis with an instantaneous sign normalization. For a matrix
    parameter :math:`\theta \in \mathbb{R}^{m \times n}` with gradient
    :math:`G_t`, the optimizer maintains a momentum :math:`M_t` and two side
    covariances :math:`L_t, R_t`:

    .. math::
       \begin{aligned}
       M_t &= \beta_1 M_{t-1} + (1 - \beta_1)\, G_t \\
       L_t &= \beta_2 L_{t-1} + (1 - \beta_2)\, G_t G_t^\top \\
       R_t &= \beta_2 R_{t-1} + (1 - \beta_2)\, G_t^\top G_t
       \end{aligned}

    Every ``inverse_steps`` steps the cached eigenbases are refreshed from the
    symmetric eigendecompositions :math:`L_t = Q_L \Lambda_L Q_L^\top` and
    :math:`R_t = Q_R \Lambda_R Q_R^\top`. The momentum is rotated into that
    basis, the sign is taken element-wise, and the result is rotated back:

    .. math::
       U_t = Q_L\, \mathrm{sign}\!\left(Q_L^\top M_t\, Q_R\right) Q_R^\top

    The update is scaled to transfer across network width by
    :math:`\gamma_t = \gamma \cdot 2 / (m + n)` for two-dimensional parameters
    and by a constant ``nonstandard_constant`` otherwise, giving
    :math:`\theta_t = \theta_{t-1} - \gamma_t U_t`. Non-matrix parameters fall
    back to a sign update :math:`U_t = \mathrm{sign}(M_t)`. An exponential
    moving average of the iterates is tracked so that
    :func:`eval` can swap in the averaged weights, which removes the parameter
    noise that large learning rates introduce.

    Note: Call :func:`eval` before validation or inference to use the averaged
    parameters, and :func:`train` to restore the raw iterates before resuming
    optimization.

    Reference: Kevin Frans, Sergey Levine, Pieter Abbeel, "A Stable Whitening
    Optimizer for Efficient Neural Network Training", 2025.
    https://arxiv.org/abs/2506.07254
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-1,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        ema_rate: float = 0.999,
        inverse_steps: int = 100,
        nonstandard_constant: float = 1e-3,
        max_dim: int = 10000,
        eps: float = 1e-30,
        maximize: bool = False,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= ema_rate <= 1.0:
            raise ValueError(f"Invalid ema_rate value: {ema_rate}")
        if not 0 < inverse_steps:
            raise ValueError(f"Invalid inverse_steps value: {inverse_steps}")
        if not 0 < max_dim:
            raise ValueError(f"Invalid max_dim value: {max_dim}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        self.maximize = maximize

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "weight_decouple": weight_decouple,
            "fixed_decay": fixed_decay,
            "ema_rate": ema_rate,
            "inverse_steps": inverse_steps,
            "max_dim": max_dim,
            "nonstandard_constant": nonstandard_constant,
            "eps": eps,
            "train_mode": True,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self) -> None:
        """Swap the raw iterates for their exponential moving average."""
        for group in self.param_groups:
            if group.get("train_mode"):
                for p in group["params"]:
                    state = self.state[p]
                    state["param_buffer"] = p.clone()
                    p.lerp_(state["ema"], weight=1.0).mul_(
                        1.0 / (1.0 - group["ema_rate"] ** group["step"])
                    )
                group["train_mode"] = False

    @torch.no_grad()
    def train(self) -> None:
        """Restore the raw iterates after a call to :func:`eval`."""
        for group in self.param_groups:
            if "train_mode" in group and not group["train_mode"]:
                for p in group["params"]:
                    state = self.state[p]
                    if "param_buffer" in state:
                        p.lerp_(state["param_buffer"], weight=1.0)
                        del state["param_buffer"]
                group["train_mode"] = True

    def init_group(self, group) -> None:
        if "step" not in group:
            group["step"] = 0

        for p in group["params"]:
            if p.grad is None:
                continue

            if p.grad.is_sparse:
                raise RuntimeError("SPlus does not support sparse gradients")
            if torch.is_complex(p):
                raise RuntimeError("SPlus does not support complex parameters")

            state = self.state[p]

            if len(state) == 0:
                state["momentum"] = torch.zeros_like(p)
                state["ema"] = torch.zeros_like(p)
                if len(p.shape) == 2:
                    state["sides"] = [
                        torch.zeros((d, d), device=p.device, dtype=p.dtype)
                        if d < group["max_dim"]
                        else None
                        for d in p.shape
                    ]
                    state["q_sides"] = [
                        torch.eye(d, device=p.device, dtype=p.dtype)
                        if d < group["max_dim"]
                        else None
                        for d in p.shape
                    ]

    @staticmethod
    def get_scaled_lr(
        shape: torch.Size,
        lr: float,
        nonstandard_constant: float,
        max_dim: int = 10000,
    ) -> float:
        scale = (
            nonstandard_constant
            if len(shape) != 2 or shape[0] > max_dim or shape[1] > max_dim
            else 2.0 / (shape[0] + shape[1])
        )
        return lr * scale

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group["step"] += 1

            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            weight_decouple = group["weight_decouple"]
            fixed_decay = group["fixed_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if self.maximize:
                    grad.neg_()

                state = self.state[p]

                scaled_lr = self.get_scaled_lr(
                    p.shape, group["lr"], group["nonstandard_constant"], group["max_dim"]
                )

                if weight_decay != 0.0:
                    if weight_decouple:
                        p.mul_(
                            1.0 - weight_decay * (1.0 if fixed_decay else scaled_lr)
                        )
                    else:
                        grad = grad.add(p, alpha=weight_decay)

                m, ema = state["momentum"], state["ema"]
                m.lerp_(grad, weight=1.0 - beta1)

                if len(p.shape) == 2:
                    sides, q_sides = state["sides"], state["q_sides"]

                    m = q_sides[0].T @ m if q_sides[0] is not None else m
                    m = m @ q_sides[1] if q_sides[1] is not None else m

                    if sides[0] is not None:
                        torch.lerp(
                            sides[0], grad @ grad.T, weight=1.0 - beta2, out=sides[0]
                        )

                    if sides[1] is not None:
                        torch.lerp(
                            sides[1], grad.T @ grad, weight=1.0 - beta2, out=sides[1]
                        )

                    update = torch.sign(m)

                    if q_sides[0] is not None:
                        update = q_sides[0] @ update

                    if q_sides[1] is not None:
                        update = update @ q_sides[1].T

                    if group["step"] == 1 or group["step"] % group["inverse_steps"] == 0:
                        if sides[0] is not None:
                            _, eig_vecs = torch.linalg.eigh(
                                sides[0].float()
                                + torch.eye(
                                    sides[0].shape[0], device=p.device
                                ).mul_(group["eps"])
                            )
                            state["q_sides"][0] = eig_vecs.to(sides[0].dtype)
                        if sides[1] is not None:
                            _, eig_vecs = torch.linalg.eigh(
                                sides[1].float()
                                + torch.eye(
                                    sides[1].shape[0], device=p.device
                                ).mul_(group["eps"])
                            )
                            state["q_sides"][1] = eig_vecs.to(sides[1].dtype)
                else:
                    update = torch.sign(m)

                p.add_(update, alpha=-scaled_lr)

                ema.lerp_(p, weight=1.0 - group["ema_rate"])

        return loss
