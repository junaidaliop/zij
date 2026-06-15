# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa0)
# Copyright (c) Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# Algorithm: APOLLO by Hanqing Zhu et al. (https://github.com/zhuhanqing/APOLLO).
# The authors' implementation is CC BY-NC 4.0 (non-commercial); this Apache-2.0
# reimplementation is vendored for license compatibility.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the APOLLO optimizer."""

import math

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["APOLLO"]


class _GaLoreProjector:
    """Low-rank gradient projector restricted to random Gaussian bases,
    the only mode APOLLO uses."""

    def __init__(self, rank, update_proj_gap, scale, projection_type):
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.projection_type = projection_type

        self.ortho_matrix = None
        self.last_update_step = -1

    def get_orthogonal_matrix(self, weights, projection_type):
        rank_sq = math.sqrt(self.rank)
        u = torch.randn(
            (weights.size(0), self.rank), device=weights.device, dtype=weights.dtype
        ).div_(rank_sq)
        vh = torch.randn(
            (self.rank, weights.size(1)), device=weights.device, dtype=weights.dtype
        ).div_(rank_sq)

        if projection_type == "right":
            return vh
        if projection_type == "left":
            return u
        return u, vh

    def update_ortho_matrix(self, x):
        is_right = x.size(0) >= x.size(1)

        if self.projection_type in ("std", "random"):
            self.ortho_matrix = self.get_orthogonal_matrix(x, "right" if is_right else "left")
        elif self.projection_type == "reverse_std":
            self.ortho_matrix = self.get_orthogonal_matrix(x, "left" if is_right else "right")
        else:
            self.ortho_matrix = self.get_orthogonal_matrix(x, self.projection_type)

    def project(self, grad, num_steps):
        update_ortho_matrix = self.ortho_matrix is None or num_steps % self.update_proj_gap == 0
        if update_ortho_matrix and num_steps != self.last_update_step:
            self.update_ortho_matrix(grad)
            self.last_update_step = num_steps

        if self.projection_type in ("std", "random"):
            if grad.shape[0] >= grad.shape[1]:
                return torch.matmul(grad, self.ortho_matrix.t())
            return torch.matmul(self.ortho_matrix.t(), grad)
        if self.projection_type == "reverse_std":
            if grad.shape[0] >= grad.shape[1]:
                return torch.matmul(self.ortho_matrix.t(), grad)
            return torch.matmul(grad, self.ortho_matrix.t())
        if self.projection_type == "right":
            return torch.matmul(grad, self.ortho_matrix.t())
        if self.projection_type == "left":
            return torch.matmul(self.ortho_matrix.t(), grad)
        return torch.matmul(torch.matmul(self.ortho_matrix[0].t(), grad), self.ortho_matrix[1].t())

    def project_back(self, low_rank_grad):
        if self.projection_type in ("std", "random"):
            return (
                torch.matmul(low_rank_grad, self.ortho_matrix)
                if low_rank_grad.shape[0] >= low_rank_grad.shape[1]
                else torch.matmul(self.ortho_matrix, low_rank_grad)
            ) * self.scale
        if self.projection_type == "reverse_std":
            return (
                torch.matmul(self.ortho_matrix, low_rank_grad)
                if low_rank_grad.shape[0] > low_rank_grad.shape[1]
                else torch.matmul(low_rank_grad, self.ortho_matrix)
            ) * self.scale
        if self.projection_type == "right":
            return torch.matmul(low_rank_grad, self.ortho_matrix) * self.scale
        if self.projection_type == "left":
            return torch.matmul(self.ortho_matrix, low_rank_grad) * self.scale
        return (
            torch.matmul(torch.matmul(self.ortho_matrix[0], low_rank_grad), self.ortho_matrix[1])
            * self.scale
        )


class APOLLO(Optimizer):
    r"""Implements APOLLO, a memory-efficient AdamW variant that replaces
    element-wise gradient scaling with channel-wise or tensor-wise factors
    estimated in a low-rank space under random projection.

    .. math::
       \begin{aligned}
            R_t &= P G_t, \qquad P_{ij} \sim \mathcal{N}(0, 1/r)               \\
            m_t &= \beta_1 m_{t-1} + (1 - \beta_1) R_t                         \\
            v_t &= \beta_2 v_{t-1} + (1 - \beta_2) R_t^2                       \\
            \tilde{R}_t &= \frac{m_t / (1 - \beta_1^t)}
                {\sqrt{v_t / (1 - \beta_2^t)} + \epsilon}                      \\
            s_j &= \frac{\|\tilde{R}_t[:, j]\|_2}{\|R_t[:, j]\|_2}
                \;\text{(channel-wise)}, \qquad
            s = \frac{\|\tilde{R}_t\|_2}{\|R_t\|_2}
                \;\text{(tensor-wise)}                                         \\
            \theta_t &= \theta_{t-1} - \eta \, \alpha \, G_t \,
                \mathrm{diag}(s) - \eta \lambda \theta_{t-1}
       \end{aligned}

    where the projection :math:`P \in \mathbb{R}^{r \times m}` is resampled
    every ``update_proj_gap`` steps and :math:`\alpha` is ``scale``. A
    norm-growth limiter caps the ratio of consecutive scaled-gradient norms
    at :math:`\gamma = 1.01`. ``scale_type='channel'`` gives APOLLO;
    ``scale_type='tensor'`` with a small rank (1 in the paper) gives
    APOLLO-Mini. With ``rank=None`` no projection is applied and the update
    reduces to AdamW.

    Note:
        Following the upstream reimplementation, the scaling factors are
        applied to the projected gradient :math:`R_t` and the update is
        mapped back through :math:`P^\top` scaled by
        :math:`\alpha^{3/2}`, rather than scaling the full-rank gradient
        :math:`G_t` directly as in Algorithm 1 of the paper.

    Reference: Hanqing Zhu, Zhenyu Zhang, Wenyan Cong, Xi Liu, Sem Park,
    Vikas Chandra, Bo Long, David Z. Pan, Zhangyang Wang, Jinwon Lee,
    "APOLLO: SGD-like Memory, AdamW-level Performance", MLSys 2025.
    https://arxiv.org/abs/2412.05270
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        scale_type: str = "tensor",
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        correct_bias: bool = True,
        eps: float = 1e-6,
        maximize: bool = False,
        rank: int | None = None,
        update_proj_gap: int = 200,
        scale: float = 1.0,
        projection_type: str = "random",
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        for i, beta in enumerate(betas):
            if not 0.0 <= beta < 1.0:
                raise ValueError(f"Invalid beta parameter at index {i}: {beta}")
        if scale_type not in ("channel", "tensor"):
            raise ValueError(f"Invalid scale_type value: {scale_type}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if rank is not None and rank < 1:
            raise ValueError(f"Invalid rank value: {rank}")
        if update_proj_gap < 1:
            raise ValueError(f"Invalid update_proj_gap value: {update_proj_gap}")
        if projection_type not in ("std", "reverse_std", "right", "left", "full", "random"):
            raise ValueError(f"Invalid projection_type value: {projection_type}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "scale_type": scale_type,
            "weight_decay": weight_decay,
            "weight_decouple": weight_decouple,
            "fixed_decay": fixed_decay,
            "correct_bias": correct_bias,
            "eps": eps,
            "maximize": maximize,
            "rank": rank,
            "update_proj_gap": update_proj_gap,
            "scale": scale,
            "projection_type": projection_type,
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

            beta1, beta2 = group["betas"]

            step_size = group["lr"]
            if group["correct_bias"]:
                bias_correction1 = 1.0 - beta1 ** group["step"]
                bias_correction2_sq = math.sqrt(1.0 - beta2 ** group["step"])
                step_size *= bias_correction2_sq / bias_correction1

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("APOLLO does not support sparse gradients")
                if group["maximize"]:
                    grad = -grad

                state = self.state[param]

                if torch.is_complex(param):
                    param = torch.view_as_real(param)
                    grad = torch.view_as_real(grad)

                projected = group["rank"] is not None and param.dim() > 1
                if projected:
                    if "projector" not in state:
                        state["projector"] = _GaLoreProjector(
                            rank=group["rank"],
                            update_proj_gap=group["update_proj_gap"],
                            scale=group["scale"],
                            projection_type=group["projection_type"],
                        )

                    grad = state["projector"].project(grad, group["step"])

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                norm_grad = exp_avg / denom
                if projected:
                    if group["scale_type"] == "channel":
                        norm_dim = 0 if norm_grad.shape[0] < norm_grad.shape[1] else 1
                        scaling_factor = torch.norm(norm_grad, dim=norm_dim) / (
                            torch.norm(grad, dim=norm_dim) + 1e-8
                        )
                        if norm_dim == 1:
                            scaling_factor = scaling_factor.unsqueeze(1)
                    else:
                        scaling_factor = torch.norm(norm_grad) / (torch.norm(grad) + 1e-8)

                    scaling_grad = grad * scaling_factor

                    scaling_grad_norm = torch.norm(scaling_grad)
                    if "scaling_grad" in state:
                        limiter = max(scaling_grad_norm / (state["scaling_grad"] + 1e-8), 1.01) / 1.01
                        scaling_grad.div_(limiter)
                        scaling_grad_norm.div_(limiter)

                    state["scaling_grad"] = scaling_grad_norm

                    norm_grad = state["projector"].project_back(
                        scaling_grad * math.sqrt(group["scale"])
                    )

                param.add_(norm_grad, alpha=-step_size)

                if group["weight_decouple"]:
                    param.mul_(
                        1.0 - group["weight_decay"] * (1.0 if group["fixed_decay"] else step_size)
                    )
                elif group["weight_decay"] > 0.0:
                    grad.add_(param, alpha=group["weight_decay"])

        return loss
