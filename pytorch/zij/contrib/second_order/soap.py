# Adapted from https://github.com/nikhilvyas/SOAP (commit a1e5535)
# Copyright (c) 2024 Nikhil Vyas. Licensed under the MIT license.
# See THIRD_PARTY_NOTICES.md for details.
"""Implementation of the SOAP optimizer."""

from itertools import chain

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["SOAP"]


class SOAP(Optimizer):
    r"""Implements SOAP, Shampoo with Adam in the Preconditioner's eigenbasis.

    SOAP keeps a Shampoo preconditioner for each tensor dimension as a running
    average of one-sided gradient outer products. Let :math:`L_t` and
    :math:`R_t` be these preconditioners for a matrix parameter, with
    eigenbases :math:`Q_L` and :math:`Q_R` refreshed every
    ``precondition_frequency`` steps. The gradient is rotated into that
    eigenbasis, Adam runs on the rotated coordinates, and the resulting update
    is rotated back:

    .. math::
       \begin{aligned}
       L_t &= \beta_2 L_{t-1} + (1 - \beta_2)\, G_t G_t^\top \\
       R_t &= \beta_2 R_{t-1} + (1 - \beta_2)\, G_t^\top G_t \\
       \tilde{G}_t &= Q_L^\top G_t Q_R \\
       m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, \tilde{G}_t \\
       v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, \tilde{G}_t^2 \\
       \theta_t &= \theta_{t-1}
           - \gamma\, \frac{\sqrt{\mathrm{bc}_2}}{\mathrm{bc}_1}\, Q_L
           \frac{m_t}{\sqrt{v_t} + \epsilon} Q_R^\top
           - \gamma\, \lambda\, \theta_{t-1}
       \end{aligned}

    where :math:`m_t` and :math:`v_t` are the Adam moments of the rotated
    gradient, :math:`\mathrm{bc}_1 = 1 - \beta_1^t` and
    :math:`\mathrm{bc}_2 = 1 - \beta_2^t` are the bias-correction terms folded
    into the scalar step size. Following the HuggingFace AdamW convention,
    :math:`\epsilon` is added to the un-bias-corrected second moment
    :math:`\sqrt{v_t}` (rather than to :math:`\sqrt{\hat{v}_t}`), which yields
    an effective denominator of
    :math:`\sqrt{\hat{v}_t} + \epsilon / \sqrt{\mathrm{bc}_2}`. The final
    term :math:`-\gamma\,\lambda\,\theta_{t-1}` is decoupled weight decay
    (:math:`\lambda` = ``weight_decay``), applied after the gradient step and
    scaled by the raw learning rate :math:`\gamma`.

    Reference: Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira,
    David Brandfonbrener, Lucas Janson, Sham Kakade,
    "SOAP: Improving and Stabilizing Shampoo using Adam for Language
    Modeling", ICLR 2025.
    https://arxiv.org/abs/2409.11321
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-3,
        betas=(0.95, 0.95),
        shampoo_beta: float = -1,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        max_precond_dim: int = 10000,
        merge_dims: bool = False,
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        data_format: str = "channels_first",
        correct_bias: bool = True,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "shampoo_beta": shampoo_beta,
            "eps": eps,
            "weight_decay": weight_decay,
            "precondition_frequency": precondition_frequency,
            "max_precond_dim": max_precond_dim,
            "merge_dims": merge_dims,
            "precondition_1d": precondition_1d,
            "normalize_grads": normalize_grads,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)
        self._data_format = data_format

    def merge_dims(self, grad, max_precond_dim):
        """Merge gradient dimensions while their product stays within
        ``max_precond_dim``."""
        assert self._data_format in ["channels_first", "channels_last"]
        if self._data_format == "channels_last" and grad.dim() == 4:
            grad = grad.permute(0, 3, 1, 2)
        shape = grad.shape
        new_shape = []

        curr_shape = 1
        for sh in shape:
            temp_shape = curr_shape * sh
            if temp_shape > max_precond_dim:
                if curr_shape > 1:
                    new_shape.append(curr_shape)
                    curr_shape = sh
                else:
                    new_shape.append(sh)
                    curr_shape = 1
            else:
                curr_shape = temp_shape

        if curr_shape > 1 or len(new_shape) == 0:
            new_shape.append(curr_shape)

        new_grad = grad.reshape(new_shape)
        return new_grad

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Arguments:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if closure is None:
            loss = None
        else:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                if "Q" not in state:
                    self.init_preconditioner(
                        grad,
                        state,
                        precondition_frequency=group["precondition_frequency"],
                        precondition_1d=group["precondition_1d"],
                        shampoo_beta=(
                            group["shampoo_beta"]
                            if group["shampoo_beta"] >= 0
                            else group["betas"][1]
                        ),
                        max_precond_dim=group["max_precond_dim"],
                        merge_dims=group["merge_dims"],
                    )
                    self.update_preconditioner(
                        grad,
                        state,
                        max_precond_dim=group["max_precond_dim"],
                        merge_dims=group["merge_dims"],
                        precondition_1d=group["precondition_1d"],
                    )
                    # First step is skipped so that we never use the current
                    # gradients in the projection.
                    continue

                # Projecting gradients to the eigenbases of Shampoo's
                # preconditioner, i.e. to the eigenbases of state['GG'].
                grad_projected = self.project(
                    grad,
                    state,
                    merge_dims=group["merge_dims"],
                    max_precond_dim=group["max_precond_dim"],
                )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad_projected, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).add_(grad_projected.square(), alpha=(1.0 - beta2))

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                exp_avg_projected = exp_avg

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** (state["step"])
                    bias_correction2 = 1.0 - beta2 ** (state["step"])
                    step_size = step_size * (bias_correction2**0.5) / bias_correction1

                # Projecting the preconditioned (by Adam) exponential moving
                # average of gradients back to the original space.
                norm_grad = self.project_back(
                    exp_avg_projected / denom,
                    state,
                    merge_dims=group["merge_dims"],
                    max_precond_dim=group["max_precond_dim"],
                )

                if group["normalize_grads"]:
                    norm_grad = norm_grad / (1e-30 + torch.mean(norm_grad**2) ** 0.5)

                p.add_(norm_grad, alpha=-step_size)

                # Decoupled weight decay, applied after the gradient step.
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

                # Update is done after the gradient step to avoid using current
                # gradients in the projection.
                self.update_preconditioner(
                    grad,
                    state,
                    max_precond_dim=group["max_precond_dim"],
                    merge_dims=group["merge_dims"],
                    precondition_1d=group["precondition_1d"],
                )

        return loss

    def init_preconditioner(
        self,
        grad,
        state,
        precondition_frequency=10,
        shampoo_beta=0.95,
        max_precond_dim=10000,
        precondition_1d=False,
        merge_dims=False,
    ):
        """Initialize the preconditioner matrices (L and R in the paper)."""
        state["GG"] = []  # Will hold the preconditioner matrices (L, R).
        if grad.dim() == 1:
            if not precondition_1d or grad.shape[0] > max_precond_dim:
                state["GG"].append([])
            else:
                state["GG"].append(
                    torch.zeros(grad.shape[0], grad.shape[0], device=grad.device)
                )
        else:
            if merge_dims:
                grad = self.merge_dims(grad, max_precond_dim)

            for sh in grad.shape:
                if sh > max_precond_dim:
                    state["GG"].append([])
                else:
                    state["GG"].append(torch.zeros(sh, sh, device=grad.device))

        state["Q"] = None  # Will hold the eigenbases of the preconditioner.
        state["precondition_frequency"] = precondition_frequency
        state["shampoo_beta"] = shampoo_beta

    def project(self, grad, state, merge_dims=False, max_precond_dim=10000):
        """Project the gradient to the eigenbases of the preconditioner."""
        original_shape = grad.shape
        if merge_dims:
            if grad.dim() == 4 and self._data_format == "channels_last":
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self.merge_dims(grad, max_precond_dim)

        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(
                    grad,
                    mat,
                    dims=[[0], [0]],
                )
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)

        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def update_preconditioner(
        self,
        grad,
        state,
        max_precond_dim=10000,
        merge_dims=False,
        precondition_1d=False,
    ):
        """Update the preconditioner matrices and eigenbases (L, R, Q_L, Q_R)."""
        if state["Q"] is not None:
            state["exp_avg"] = self.project_back(
                state["exp_avg"],
                state,
                merge_dims=merge_dims,
                max_precond_dim=max_precond_dim,
            )
        if grad.dim() == 1:
            if precondition_1d and grad.shape[0] <= max_precond_dim:
                state["GG"][0].lerp_(
                    grad.unsqueeze(1) @ grad.unsqueeze(0), 1 - state["shampoo_beta"]
                )
        else:
            if merge_dims:
                new_grad = self.merge_dims(grad, max_precond_dim)
                for idx, sh in enumerate(new_grad.shape):
                    if sh <= max_precond_dim:
                        outer_product = torch.tensordot(
                            new_grad,
                            new_grad,
                            dims=[
                                [*chain(range(idx), range(idx + 1, len(new_grad.shape)))]
                            ]
                            * 2,
                        )
                        state["GG"][idx].lerp_(outer_product, 1 - state["shampoo_beta"])
            else:
                for idx, sh in enumerate(grad.shape):
                    if sh <= max_precond_dim:
                        outer_product = torch.tensordot(
                            grad,
                            grad,
                            # Contracts across all dimensions except for idx.
                            dims=[
                                [*chain(range(idx), range(idx + 1, len(grad.shape)))]
                            ]
                            * 2,
                        )
                        state["GG"][idx].lerp_(outer_product, 1 - state["shampoo_beta"])

        if state["Q"] is None:
            state["Q"] = self.get_orthogonal_matrix(state["GG"])
        if state["step"] > 0 and state["step"] % state["precondition_frequency"] == 0:
            state["Q"] = self.get_orthogonal_matrix_QR(state, max_precond_dim, merge_dims)

        if state["step"] > 0:
            state["exp_avg"] = self.project(
                state["exp_avg"],
                state,
                merge_dims=merge_dims,
                max_precond_dim=max_precond_dim,
            )

    def project_back(self, grad, state, merge_dims=False, max_precond_dim=10000):
        """Project the gradient back to the original space."""
        original_shape = grad.shape
        if merge_dims:
            if self._data_format == "channels_last" and grad.dim() == 4:
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self.merge_dims(grad, max_precond_dim)
        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(
                    grad,
                    mat,
                    dims=[[0], [1]],
                )
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)

        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def get_orthogonal_matrix(self, mat):
        """Compute the eigenbases of the preconditioner with
        :func:`torch.linalg.eigh`."""
        matrix = []
        for m in mat:
            if len(m) == 0:
                matrix.append([])
                continue
            if m.data.dtype != torch.float:
                float_data = False
                original_type = m.data.dtype
                original_device = m.data.device
                matrix.append(m.data.float())
            else:
                float_data = True
                matrix.append(m.data)

        final = []
        for m in matrix:
            if len(m) == 0:
                final.append([])
                continue
            try:
                _, Q = torch.linalg.eigh(
                    m + 1e-30 * torch.eye(m.shape[0], device=m.device)
                )
            except Exception:
                _, Q = torch.linalg.eigh(
                    m.to(torch.float64)
                    + 1e-30 * torch.eye(m.shape[0], device=m.device)
                )
                Q = Q.to(m.dtype)
            Q = torch.flip(Q, [1])

            if not float_data:
                Q = Q.to(original_device).type(original_type)
            final.append(Q)
        return final

    def get_orthogonal_matrix_QR(self, state, max_precond_dim=10000, merge_dims=False):
        """Compute the eigenbases of the preconditioner with one round of power
        iteration followed by :func:`torch.linalg.qr`."""
        precond_list = state["GG"]
        orth_list = state["Q"]

        matrix = []
        orth_matrix = []
        for m, o in zip(precond_list, orth_list):
            if len(m) == 0:
                matrix.append([])
                orth_matrix.append([])
                continue
            if m.data.dtype != torch.float:
                float_data = False
                original_type = m.data.dtype
                original_device = m.data.device
                matrix.append(m.data.float())
                orth_matrix.append(o.data.float())
            else:
                float_data = True
                matrix.append(m.data.float())
                orth_matrix.append(o.data.float())

        orig_shape = state["exp_avg_sq"].shape
        if self._data_format == "channels_last" and len(orig_shape) == 4:
            permuted_shape = state["exp_avg_sq"].permute(0, 3, 1, 2).shape
        if merge_dims:
            exp_avg_sq = self.merge_dims(state["exp_avg_sq"], max_precond_dim)
        else:
            exp_avg_sq = state["exp_avg_sq"]

        final = []
        for ind, (m, o) in enumerate(zip(matrix, orth_matrix)):
            if len(m) == 0:
                final.append([])
                continue
            est_eig = torch.diag(o.T @ m @ o)
            sort_idx = torch.argsort(est_eig, descending=True)
            exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
            o = o[:, sort_idx]
            power_iter = m @ o
            Q, _ = torch.linalg.qr(power_iter)

            if not float_data:
                Q = Q.to(original_device).type(original_type)
            final.append(Q)

        if merge_dims:
            if self._data_format == "channels_last" and len(orig_shape) == 4:
                exp_avg_sq = exp_avg_sq.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                exp_avg_sq = exp_avg_sq.reshape(orig_shape)

        state["exp_avg_sq"] = exp_avg_sq
        return final
