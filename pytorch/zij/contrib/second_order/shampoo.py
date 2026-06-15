# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa0)
# Copyright (c) Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# A PyTorch port of the original JAX Shampoo from google-research/google-research.
# See THIRD_PARTY_NOTICES.md for details.
"""Implementation of the Shampoo optimizer."""

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["Shampoo"]


def _compute_power_svd(matrix: torch.Tensor, power: float) -> torch.Tensor:
    """Compute ``matrix^{-1/power}`` through a singular value decomposition."""
    u, s, vh = torch.linalg.svd(matrix.to(torch.float32), full_matrices=False)
    s.pow_(-1.0 / power)
    diag = s.diag() if matrix.dim() == 2 else s.diag_embed()
    return (u @ diag @ vh).to(matrix.dtype)


class Shampoo(Optimizer):
    r"""Implements Shampoo, preconditioned stochastic tensor optimization.

    For a matrix parameter :math:`W` with gradient :math:`G_t`, Shampoo keeps a
    left preconditioner :math:`L_t` over the rows and a right preconditioner
    :math:`R_t` over the columns, each accumulated from the gradient outer
    products, and conditions the update on both sides:

    .. math::
       \begin{aligned}
       L_t &= L_{t-1} + G_t G_t^\top \\
       R_t &= R_{t-1} + G_t^\top G_t \\
       W_{t+1} &= W_t - \eta\, L_t^{-1/2}\, G_t\, R_t^{-1/2}
       \end{aligned}

    For a general order-:math:`k` tensor a preconditioner is maintained for
    every dimension by contracting the gradient over the remaining axes, and the
    inverse root applied per dimension uses exponent :math:`-1/k`.

    Note: the original paper (Algorithm 1, matrix case) applies the exponent
    :math:`-1/4` to each preconditioner, giving
    :math:`W_{t+1} = W_t - \eta\, L_t^{-1/4} G_t R_t^{-1/4}`. This
    implementation instead raises each preconditioner to :math:`-1/k` for an
    order-:math:`k` tensor (so :math:`-1/2` for matrices), and recomputes the
    inverse roots every ``preconditioning_compute_steps`` steps.

    Reference: Vineet Gupta, Tomer Koren, Yoram Singer, "Shampoo:
    Preconditioned Stochastic Tensor Optimization", ICML 2018.
    https://arxiv.org/abs/1802.09568
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        preconditioning_compute_steps: int = 1,
        matrix_eps: float = 1e-6,
        maximize: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Learning rate {lr} must be non-negative")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"Momentum {momentum} must be in the range [0,1]")
        if weight_decay < 0.0:
            raise ValueError(f"Weight decay {weight_decay} must be non-negative")
        if preconditioning_compute_steps < 1:
            raise ValueError(
                f"preconditioning_compute_steps {preconditioning_compute_steps} must be positive"
            )
        if matrix_eps < 0.0:
            raise ValueError(f"matrix_eps {matrix_eps} must be non-negative")

        self.preconditioning_compute_steps = preconditioning_compute_steps
        self.maximize = maximize

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            weight_decouple=weight_decouple,
            fixed_decay=fixed_decay,
            matrix_eps=matrix_eps,
        )
        super().__init__(params, defaults)

    def init_group(self, group):
        if "step" not in group:
            group["step"] = 0

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("Shampoo does not support sparse gradients")
            if torch.is_complex(p):
                raise RuntimeError("Shampoo does not support complex parameters")

            state = self.state[p]
            if len(state) == 0:
                if group["momentum"] > 0.0:
                    state["momentum_buffer"] = grad.clone()
                for dim_id, dim in enumerate(grad.size()):
                    state[f"pre_cond_{dim_id}"] = group["matrix_eps"] * torch.eye(
                        dim, out=grad.new(dim, dim)
                    )
                    state[f"inv_pre_cond_{dim_id}"] = grad.new(dim, dim).zero_()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group["step"] += 1

            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            weight_decouple = group["weight_decouple"]
            fixed_decay = group["fixed_decay"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if self.maximize:
                    grad.neg_()

                state = self.state[p]

                if momentum > 0.0:
                    grad.mul_(1.0 - momentum).add_(state["momentum_buffer"], alpha=momentum)

                if weight_decouple:
                    p.mul_(1.0 - weight_decay * (1.0 if fixed_decay else lr))
                elif weight_decay > 0.0:
                    grad.add_(p, alpha=weight_decay)

                order = grad.ndimension()
                original_size = grad.size()
                for dim_id, dim in enumerate(grad.size()):
                    pre_cond = state[f"pre_cond_{dim_id}"]
                    inv_pre_cond = state[f"inv_pre_cond_{dim_id}"]

                    grad = grad.transpose_(0, dim_id).contiguous()
                    transposed_size = grad.size()

                    grad = grad.view(dim, -1)
                    grad_t = grad.t()

                    pre_cond.add_(grad @ grad_t)
                    if group["step"] % self.preconditioning_compute_steps == 0:
                        inv_pre_cond.copy_(_compute_power_svd(pre_cond, order))

                    if dim_id == order - 1:
                        grad = grad_t @ inv_pre_cond
                        grad = grad.view(original_size)
                    else:
                        grad = inv_pre_cond @ grad
                        grad = grad.view(transposed_size)

                state["momentum_buffer"] = grad

                p.add_(grad, alpha=-lr)

        return loss
