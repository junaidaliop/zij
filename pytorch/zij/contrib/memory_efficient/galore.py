# Adapted from https://github.com/jiaweizzhao/GaLore (commit 2cc66f8)
# Copyright (c) 2024 Jiawei Zhao. Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
"""Implementation of the GaLore projector and the GaLoreAdamW optimizer."""

import math
from typing import Callable, Iterable, Tuple

import torch
from torch import nn

from ...core.optimizer import Optimizer

__all__ = ["GaLoreAdamW", "GaLoreProjector"]


class GaLoreProjector:
    """Projects gradients onto a low-rank subspace obtained by truncated SVD."""

    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0, proj_type='std'):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type

    def project(self, full_rank_grad, iter):
        if self.proj_type == 'std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t().to(full_rank_grad.device.type))
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                low_rank_grad = torch.matmul(self.ortho_matrix.t().to(full_rank_grad.device.type), full_rank_grad)
        elif self.proj_type == 'reverse_std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                low_rank_grad = torch.matmul(self.ortho_matrix.t().to(full_rank_grad.device.type), full_rank_grad)
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t().to(full_rank_grad.device.type))
        elif self.proj_type == 'right':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t().to(full_rank_grad.device.type))
        elif self.proj_type == 'left':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
            low_rank_grad = torch.matmul(self.ortho_matrix.t().to(full_rank_grad.device.type), full_rank_grad)
        elif self.proj_type == 'full':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='full')
            low_rank_grad = torch.matmul(self.ortho_matrix[0].t().to(full_rank_grad.device.type), full_rank_grad) @ self.ortho_matrix[1].t().to(full_rank_grad.device.type)

        return low_rank_grad

    def project_back(self, low_rank_grad):
        if self.proj_type == 'std':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix.to(low_rank_grad.device.type))
            else:
                full_rank_grad = torch.matmul(self.ortho_matrix.to(low_rank_grad.device.type), low_rank_grad)
        elif self.proj_type == 'reverse_std':
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]:  # note this is different from std
                full_rank_grad = torch.matmul(self.ortho_matrix.to(low_rank_grad.device.type), low_rank_grad)
            else:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix.to(low_rank_grad.device.type))
        elif self.proj_type == 'right':
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix.to(low_rank_grad.device.type))
        elif self.proj_type == 'left':
            full_rank_grad = torch.matmul(self.ortho_matrix.to(low_rank_grad.device.type), low_rank_grad)
        elif self.proj_type == 'full':
            full_rank_grad = torch.matmul(self.ortho_matrix[0].to(low_rank_grad.device.type), low_rank_grad) @ self.ortho_matrix[1].to(low_rank_grad.device.type)

        return full_rank_grad * self.scale

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
        if type == 'right':
            B = Vh[:rank, :]
            if not float_data:
                B = B.to(original_device).type(original_type)
            return B
        elif type == 'left':
            A = U[:, :rank]
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        elif type == 'full':
            A = U[:, :rank]
            B = Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
                B = B.to(original_device).type(original_type)
            return [A, B]
        else:
            raise ValueError('type should be left, right or full')


class GaLoreAdamW(Optimizer):
    r"""Implements GaLoreAdamW, AdamW with gradient low-rank projection.

    .. math::
       \begin{aligned}
       &P_t = U[:, {:}r] \quad \text{where} \quad
           U S V^\top = \mathrm{SVD}(g_t)
           \quad \text{(recomputed every $T$ steps)}                       \\
       &r_t = P_t^\top g_t                                                 \\
       &m_t = \beta_1 m_{t-1} + (1 - \beta_1)\, r_t                        \\
       &v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, r_t^2                      \\
       &\eta_t = \eta\, \sqrt{1 - \beta_2^t} / (1 - \beta_1^t)            \\
       &\tilde{g}_t = \alpha\, P_t\, m_t / (\sqrt{v_t} + \epsilon)         \\
       &\theta_{t+1} = (1 - \eta\lambda)\,(\theta_t - \eta_t\, \tilde{g}_t)
       \end{aligned}

    where :math:`r` is the projection rank, :math:`T` the subspace change
    frequency (``update_proj_gap``), :math:`\alpha` the ``scale`` factor, and
    :math:`\lambda` the decoupled weight decay, applied after the gradient
    step as upstream does. Bias correction is folded into the step size
    :math:`\eta_t`, the formulation the official implementation inherits
    from the transformers AdamW. The Adam statistics
    :math:`m_t, v_t` live in the rank-:math:`r` subspace, which is what saves
    the optimizer memory. The paper states the update for a matrix with
    :math:`m \le n` and a left projector; this implementation picks the
    projector side from the gradient shape so the smaller factor is kept.

    Note:
        Projection is enabled per parameter group: groups carrying ``rank``,
        ``update_proj_gap``, ``scale``, and ``proj_type`` keys are projected
        (2D parameters only), all other groups get plain AdamW. The upstream
        tensor projector for dim > 2 parameters needs tensorly and is not
        vendored.

    Reference: Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang,
    Anima Anandkumar, Yuandong Tian,
    "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection",
    ICML 2024.
    https://arxiv.org/abs/2403.03507
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

                if "dim" not in group:
                    group["dim"] = 2

                # GaLore Projection
                if "rank" in group:
                    if "projector" not in state:
                        if group["dim"] > 2:
                            raise ValueError(
                                "GaLoreAdamW projects 2D parameters only; the upstream "
                                "tensor projector for dim > 2 needs tensorly and is not vendored"
                            )
                        state["projector"] = GaLoreProjector(group["rank"], update_proj_gap=group["update_proj_gap"], scale=group["scale"], proj_type=group["proj_type"])
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

                # GaLore Projection Back
                if "rank" in group:
                    norm_grad = state["projector"].project_back(norm_grad)

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
