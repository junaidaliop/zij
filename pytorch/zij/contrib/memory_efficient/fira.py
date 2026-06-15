# Adapted from https://github.com/xichen-fy/Fira (commit 5af6a98)
# Copyright (c) 2024 Xi Chen (xichen-fy). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
"""Implementation of the Fira gradient projector and the FiraAdamW optimizer."""

import math
from typing import Callable, Iterable, Tuple

import torch
from torch import nn

from ...core.optimizer import Optimizer

__all__ = ["FiraAdamW", "GradientProjector"]


class GradientProjector:
    """Projects gradients onto a low-rank subspace obtained by truncated SVD."""

    def __init__(
        self, rank, verbose=False, update_proj_gap=200, alpha=1.0, proj_type="std"
    ):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.alpha = alpha
        self.ortho_matrix = None
        self.proj_type = proj_type

    def project(self, full_rank_grad, iter):
        if self.proj_type == "std":
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type="right"
                    )
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type="left"
                    )
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == "reverse_std":
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type="left"
                    )
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type="right"
                    )
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        elif self.proj_type == "right":
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type="right"
                )
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        elif self.proj_type == "left":
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type="left"
                )
            low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == "full":
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type="full"
                )
            low_rank_grad = (
                torch.matmul(self.ortho_matrix[0].t(), full_rank_grad)
                @ self.ortho_matrix[1].t()
            )

        return low_rank_grad

    def project_back(self, low_rank_grad):
        if self.proj_type == "std":
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
            else:
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == "reverse_std":
            if (
                low_rank_grad.shape[0] <= low_rank_grad.shape[1]
            ):  # note this is different from std
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
            else:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == "right":
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == "left":
            full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == "full":
            full_rank_grad = (
                torch.matmul(self.ortho_matrix[0], low_rank_grad) @ self.ortho_matrix[1]
            )

        return full_rank_grad * self.alpha

    # svd decomposition
    def get_orthogonal_matrix(self, weights, rank, type):
        module_params = weights

        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data

        U, s, Vh = torch.linalg.svd(matrix, full_matrices=False)

        # make the smaller matrix always to be orthogonal matrix
        if type == "right":
            B = Vh[:rank, :]
            if not float_data:
                B = B.to(original_device).type(original_type)
            return B
        elif type == "left":
            A = U[:, :rank]
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        elif type == "full":
            A = U[:, :rank]
            B = Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
                B = B.to(original_device).type(original_type)
            return [A, B]
        else:
            raise ValueError("type should be left, right or full")


class FiraAdamW(Optimizer):
    r"""Implements FiraAdamW, AdamW with full-rank updates under a GaLore-style
    low-rank optimizer-memory budget.

    .. math::
       \begin{aligned}
       &P_t = U[:, {:}r] \quad \text{where} \quad
           U S V^\top = \mathrm{SVD}(g_t)
           \quad \text{(recomputed every $T$ steps)}                       \\
       &r_t = P_t^\top g_t                                                 \\
       &m_t = \beta_1 m_{t-1} + (1 - \beta_1)\, r_t                        \\
       &v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, r_t^2                      \\
       &\eta_t = \eta\, \sqrt{1 - \beta_2^t} / (1 - \beta_1^t)             \\
       &\psi_t = m_t / (\sqrt{v_t} + \epsilon)                             \\
       &(\phi_t)_i = \lVert (\psi_t)_{:,i} \rVert_2
           \, / \, \lVert (r_t)_{:,i} \rVert_2                             \\
       &S_t = \phi_t \odot (g_t - \alpha P_t r_t)                          \\
       &S_t \leftarrow \gamma\, S_t\, \lVert S_{t-1} \rVert
           / \lVert S_t \rVert
           \quad \text{if } \lVert S_t \rVert > \gamma \lVert S_{t-1} \rVert \\
       &\theta_{t+1} = (1 - \eta\lambda)\,
           (\theta_t - \eta_t\, (\alpha P_t \psi_t + S_t))
       \end{aligned}

    where :math:`r` is the projection rank, :math:`T` the subspace change
    frequency (``update_proj_gap``), :math:`\alpha` the ``alpha`` scale
    factor, :math:`\gamma = 1.01` the norm-growth limit, and :math:`\lambda`
    the decoupled weight decay, applied after the gradient step as upstream
    does. Bias correction is folded into the step size :math:`\eta_t`. The
    Adam statistics :math:`m_t, v_t` live in the rank-:math:`r` subspace, as
    in GaLore; the residual gradient :math:`g_t - \alpha P_t r_t` outside the
    subspace is applied with the norm-based scaling :math:`\phi_t`, one
    factor per column of :math:`r_t` (per row when the gradient is projected
    from the right), so the full-rank direction is trained without full-rank
    optimizer state. The norm-growth limiter caps the step-to-step growth of
    :math:`\lVert S_t \rVert` at :math:`\gamma` to suppress loss spikes.

    Note:
        Projection is enabled per parameter group: groups carrying ``rank``,
        ``update_proj_gap``, ``alpha``, and ``proj_type`` keys are projected
        (2D parameters only), all other groups get plain AdamW.

    Reference: Xi Chen, Kaituo Feng, Changsheng Li, Xunhao Lai, Xiangyu Yue,
    Ye Yuan, Guoren Wang,
    "Fira: Can We Achieve Full-rank Training of LLMs Under Low-rank
    Constraint?", NeurIPS 2025.
    https://arxiv.org/abs/2410.01623
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # Gradient Projection
                if "rank" in group:
                    if "projector" not in state:
                        if p.dim() != 2:
                            raise ValueError("FiraAdamW projects 2D parameters only")
                        state["projector"] = GradientProjector(group["rank"], update_proj_gap=group["update_proj_gap"], alpha=group["alpha"], proj_type=group["proj_type"])
                    grad = state["projector"].project(grad, state["step"])

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # compute norm gradient
                norm_grad = exp_avg / denom

                # Gradient Projection Back
                if "rank" in group:
                    # Norm-Based Scaling
                    subgrad = state["projector"].project_back(grad)
                    norm_dim = 0 if norm_grad.shape[0] < norm_grad.shape[1] else 1
                    scaling_factor = (
                        torch.norm(norm_grad, dim=norm_dim) /
                        (torch.norm(grad, dim=norm_dim) + 1e-8)
                    )
                    if norm_dim == 1:
                        scaling_factor = scaling_factor.unsqueeze(1)
                    scaling_grad = (p.grad - subgrad) * scaling_factor

                    # Norm-Growth Limiter
                    if "scaling_grad" in state:
                        scaling_grad_norm = torch.norm(scaling_grad)
                        limiter = max(
                                scaling_grad_norm /
                                (state["scaling_grad"] + 1e-8),
                                1.01,
                            ) / 1.01
                        scaling_grad = scaling_grad / limiter
                        state["scaling_grad"] = scaling_grad_norm / limiter
                    else:
                        state["scaling_grad"] = torch.norm(scaling_grad)

                    norm_grad = state["projector"].project_back(norm_grad) + scaling_grad

                p.add_(norm_grad, alpha=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss
