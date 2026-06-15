# Adapted from https://github.com/IST-DASLab/LDAdam (commit f3c5daf)
# Copyright (c) 2024 Thomas Robert (IST-DASLab). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
"""Implementation of the LDAdam optimizer."""

import torch

from ...core.optimizer import Optimizer

__all__ = ["LDAdamW"]


class _LowRankProjector:
    def __init__(self, rank, proj_type='std'):
        self.rank = rank
        self.proj_type = proj_type
        self.ortho_matrix = None

    def project(self, full_rank_grad):
        if self.proj_type == 'right':
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        elif self.proj_type == 'left':
            low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)

        return low_rank_grad

    def project_back(self, low_rank_grad):
        if self.proj_type == 'right':
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == 'left':
            full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)

        return full_rank_grad

    # svd decomposition
    def get_orthogonal_matrix_svd(self, grad, svd_lowrank=False):
        matrix = grad.data.clone()

        if matrix.dtype != torch.float:  # torch.linalg.svd doesn't support half precision types such as torch.bfloat16
            float_data = False
            original_type = matrix.dtype
            original_device = matrix.device
            matrix = matrix.float()
        else:
            float_data = True

        if svd_lowrank:
            U, s, V = torch.svd_lowrank(matrix, q=self.rank + 2, niter=1)  # q a slightly overestimated rank of A
            Vh = V.t()
        else:
            U, s, Vh = torch.linalg.svd(matrix, full_matrices=False)

        if self.proj_type == 'right':
            ortho_matrix = Vh[:self.rank, :]
        elif self.proj_type == 'left':
            ortho_matrix = U[:, :self.rank]

        if not float_data:
            ortho_matrix = ortho_matrix.to(device=original_device, dtype=original_type)

        self.ortho_matrix = ortho_matrix

    def power_iteration(self, matrix, init, intermediate_orthogonalization=False):
        if self.proj_type == 'right':
            U = matrix @ init.t()
            if intermediate_orthogonalization:  # not necessary for computing right singular vectors
                U = _gram_schmidt(U)
            projection_map = matrix.t() @ U
            del U

            projection_map = _gram_schmidt(projection_map)
            self.ortho_matrix = projection_map.t()

        elif self.proj_type == 'left':
            V = matrix.t() @ init
            if intermediate_orthogonalization:  # not necessary for computing left singular vectors
                V = _gram_schmidt(V)
            projection_map = matrix @ V
            del V

            projection_map = _gram_schmidt(projection_map)
            self.ortho_matrix = projection_map


def _gram_schmidt(matrix):
    original_type = matrix.dtype  # torch.linalg.qr doesn't support half precision types such as torch.bfloat16
    matrix, _ = torch.linalg.qr(matrix.to(dtype=torch.float32))
    matrix = matrix.to(dtype=original_type)

    return matrix


class LDAdamW(Optimizer):
    r"""Implements LDAdam, an Adam variant that keeps its moment estimates in an
    adaptively tracked low-dimensional subspace.

    With :math:`P_t` the rank-:math:`r` orthogonal projection tracked by one
    round of block power iteration per step (initialized from a truncated SVD
    of the first gradient), :math:`Q_t = P_t^\top P_{t-1}` the change-of-basis
    matrix, and hats denoting Adam bias correction:

    .. math::
       \begin{aligned}
       a_t &= g_t + e_t \\
       b_t &= \rho\, P_{t-1} \hat{m}_{t-1} + (1 - \rho)\, a_t \\
       P_t &= \mathrm{QR}\big(b_t b_t^\top P_{t-1}\big), \qquad
       \tilde{g}_t = P_t^\top a_t \\
       \tilde{m}_{t-1} &= Q_t m_{t-1} \\
       \tilde{v}_{t-1} &= (1 - \beta_2^{t-1})\,
           \big| (Q_t \circ Q_t)\big(\hat{v}_{t-1} - \hat{m}_{t-1}^{\,2}\big)
           + (Q_t \hat{m}_{t-1})^2 \big| \\
       m_t &= \beta_1 \tilde{m}_{t-1} + (1 - \beta_1)\, \tilde{g}_t \\
       v_t &= \beta_2 \tilde{v}_{t-1} + (1 - \beta_2)\, \tilde{g}_t^{\,2} \\
       \theta_{t+1} &= (1 - \eta\lambda)\, \theta_t
           - \eta\, P_t\, \hat{m}_t / \big(\sqrt{\hat{v}_t} + \epsilon\big) \\
       e_{t+1} &= a_t - P_t \tilde{g}_t
           + \tfrac{\beta_1}{1 - \beta_1}\big(P_{t-1} m_{t-1} - P_t \tilde{m}_{t-1}\big)
       \end{aligned}

    where :math:`\rho` interpolates between the past momentum and the fresh
    gradient when adapting the subspace, :math:`\lambda` is the decoupled
    weight decay, and :math:`e_t` is the generalized error feedback buffer
    that captures both gradient and optimizer-state compression error. The
    equations are written for left projection; tall matrices (more rows than
    columns) are projected from the right, wide matrices from the left,
    mirroring the update.

    Note:
        Low-rank compression applies to 2D parameters only. Group parameters
        and set ``enable_lowrank=False`` on groups holding biases, norms, and
        embeddings, which then take plain AdamW steps; the upstream
        experiments enable low-rank only for linear-layer weights. With
        ``error_feedback=True``, gradients of low-rank parameters double as
        the error buffer: ``zero_grad()`` leaves them in place, and gradient
        accumulation adds onto the buffer as the algorithm expects.

    Reference: Thomas Robert, Mher Safaryan, Ionut-Vlad Modoranu, Dan Alistarh,
    "LDAdam: Adaptive Optimization from Low-Dimensional Gradient Statistics",
    ICLR 2025.
    https://arxiv.org/abs/2410.16103
    """

    def __init__(self,
        params,
        lr: float = 0.001,
        betas: tuple[float, float] = (0.908, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        rank: int = 16,
        rho: float = 0.908,
        proj_type: str = 'std',
        proj_method: str = 'power_iteration',
        error_feedback: bool = True,
        enable_lowrank: bool = True,
    ):

        # Sanity check
        if not isinstance(lr, (int, float)) or lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not isinstance(betas, tuple) or len(betas) != 2 or not all(isinstance(beta, (int, float)) for beta in betas) or not all(0.0 <= beta < 1.0 for beta in betas):
            raise ValueError("Invalid betas: {}".format(betas))
        if not isinstance(eps, (int, float)) or eps <= 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not isinstance(weight_decay, (int, float)) or weight_decay < 0.0:
            raise ValueError("Invalid weight decay value: {}".format(weight_decay))
        if not isinstance(rank, int) or rank <= 0:
            raise ValueError("Invalid rank value: {}".format(rank))
        if not isinstance(rho, (int, float)) or not 0.0 <= rho < 1.0:
            raise ValueError("Invalid rho value: {}".format(rho))
        if proj_type not in ['std', 'left', 'right', 'reverse_std']:
            raise ValueError("Invalid projection type: {}".format(proj_type))
        if not isinstance(proj_method, str) or proj_method not in ['svd', 'svd_lowrank', 'power_iteration']:
            raise ValueError("Invalid projection method: {}".format(proj_method))
        if not isinstance(error_feedback, bool):
            raise ValueError("Invalid error feedback value: {}".format(error_feedback))
        if not isinstance(enable_lowrank, bool):
            raise ValueError("Invalid enable lowrank value: {}".format(enable_lowrank))

        # Construct optimizer
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps,
                        rank=rank, rho=rho, proj_type=proj_type,
                        proj_method=proj_method, error_feedback=error_feedback,
                        enable_lowrank=enable_lowrank)
        super().__init__(params, defaults)

        # Default hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.error_feedback = error_feedback

        # Initialize optimizer states
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue

                st = self.state[p]

                # AdamW hyperparameters
                st['lr'] = group.get('lr', lr)
                st['beta1'] = group.get('beta1', betas[0])
                st['beta2'] = group.get('beta2', betas[1])
                st['weight_decay'] = group.get('weight_decay', weight_decay)
                st['eps'] = group.get('eps', eps)

                if not group['enable_lowrank']:
                    st['m'] = torch.zeros_like(p)
                    st['v'] = torch.zeros_like(p)
                    continue

                # LDAdamW hyperparameters
                st['rho'] = group.get('rho', rho)
                st['rank'] = group.get('rank', rank)

                layer_shape = p.shape
                if len(layer_shape) != 2:
                    raise ValueError(
                        "Low-rank compression expects 2D parameters, got shape {}; "
                        "put this parameter in a group with enable_lowrank=False".format(tuple(layer_shape)))
                group_proj_type = group.get('proj_type', proj_type)
                st['left_proj'], st['right_proj'] = False, False
                if group_proj_type == 'right' or (layer_shape[0] > layer_shape[1] and group_proj_type == 'std') or (layer_shape[0] < layer_shape[1] and group_proj_type == 'reverse_std'):
                    st['right_proj'] = True
                    st['previous_projector'] = _LowRankProjector(rank=st['rank'], proj_type='right')
                    lowdim_shape = (layer_shape[0], st['rank'])
                    if st['rank'] > layer_shape[1]:
                        raise ValueError("For right projection, rank cannot be greater than the number of columns in the weight matrix")
                elif group_proj_type == 'left' or (layer_shape[0] <= layer_shape[1] and group_proj_type == 'std') or (layer_shape[0] >= layer_shape[1] and group_proj_type == 'reverse_std'):
                    st['left_proj'] = True
                    st['previous_projector'] = _LowRankProjector(rank=st['rank'], proj_type='left')
                    lowdim_shape = (st['rank'], layer_shape[1])
                    if st['rank'] > layer_shape[0]:
                        raise ValueError("For left projection, rank cannot be greater than the number of rows in the weight matrix")

                st['m'] = torch.zeros(lowdim_shape, device=p.device, dtype=p.dtype)
                st['v'] = torch.zeros(lowdim_shape, device=p.device, dtype=p.dtype)

                group_proj_method = group.get('proj_method', proj_method)
                st['use_svd'], st['use_svd_lowrank'], st['use_poweriteration'] = False, False, False
                if group_proj_method == 'svd' or group_proj_method == 'svd_lowrank':
                    st['use_svd'] = True
                if group_proj_method == 'svd_lowrank':
                    st['use_svd_lowrank'] = True
                if group_proj_method == 'power_iteration':
                    st['use_poweriteration'] = True

                st['error_feedback'] = group.get('error_feedback', error_feedback)

        self.completed_steps = 0

    @torch.no_grad()
    def step(self, closure=None):
        self._update_lr_wd()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group['enable_lowrank']:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    self.ldadamw_step(p)
            else:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    self.adamw_step(p)

        self.completed_steps += 1

        return loss

    @torch.no_grad()
    def adamw_step(self, p):
        completed_steps = self.completed_steps

        st = self.state[p]
        grad = p.grad

        # AdamW hyperparameters
        lr = st['lr']
        wd = st['weight_decay']
        beta1 = st['beta1']
        beta2 = st['beta2']
        eps = st['eps']

        # Adaptive optimization step
        st['m'].mul_(beta1)
        st['m'].add_(grad, alpha=(1 - beta1))

        st['v'].mul_(beta2)
        st['v'].addcmul_(grad, grad, value=(1 - beta2))

        ### MODEL UPDATE
        descent_direction = st['v'].div(1 - beta2**(completed_steps + 1))
        descent_direction.sqrt_()
        descent_direction.add_(eps)

        descent_direction.reciprocal_()
        descent_direction.mul_(st['m'])
        descent_direction.div_(1 - beta1**(completed_steps + 1))

        p.mul_(1 - lr * wd)  # decoupled weight decay
        p.add_(descent_direction, alpha=-lr)

        del descent_direction

    @torch.no_grad()
    def ldadamw_step(self, p):
        completed_steps = self.completed_steps

        st = self.state[p]
        grad = p.grad

        # AdamW hyperparameters
        lr = st['lr']
        wd = st['weight_decay']
        beta1 = st['beta1']
        beta2 = st['beta2']
        eps = st['eps']

        # LDAdamW hyperparameters
        rho = st['rho']
        rank = st['rank']
        left_proj = st['left_proj']
        right_proj = st['right_proj']
        use_svd = st['use_svd']
        use_svd_lowrank = st['use_svd_lowrank']
        use_poweriteration = st['use_poweriteration']

        ### LEARNING SUBSPACE ADAPTATION
        if left_proj:
            projector = _LowRankProjector(rank=rank, proj_type='left')
        elif right_proj:
            projector = _LowRankProjector(rank=rank, proj_type='right')

        previous_projector = st['previous_projector']

        if completed_steps == 0:
            projector.get_orthogonal_matrix_svd(grad, svd_lowrank=use_svd_lowrank)  # init power iteration process with SVD
            previous_projector.ortho_matrix = torch.zeros_like(projector.ortho_matrix)
        else:
            b = previous_projector.project_back(st['m'])
            b.div_(1 - beta1**completed_steps)
            b.mul_(rho)
            b.add_(grad, alpha=(1 - rho))
            if use_svd:
                projector.get_orthogonal_matrix_svd(b, svd_lowrank=use_svd_lowrank)
            elif use_poweriteration:
                projector.power_iteration(b, init=previous_projector.ortho_matrix)

        lowdim_grad = projector.project(grad)

        ### ERROR BUFFER LOADING - gradient compression
        if st['error_feedback']:
            lowrank_grad = projector.project_back(lowdim_grad)
            grad.sub_(lowrank_grad)  # store error in grad tensor
            del lowrank_grad

        ### OPTIMIZER STATES PROJECTION-AWARE UPDATE - gradient first-order statistic
        if left_proj:
            mat_change_of_subspace = projector.ortho_matrix.t() @ previous_projector.ortho_matrix
            lowdim_updated_momentum = mat_change_of_subspace @ st['m']

        elif right_proj:
            mat_change_of_subspace = previous_projector.ortho_matrix @ projector.ortho_matrix.t()
            lowdim_updated_momentum = st['m'] @ mat_change_of_subspace

        ### GENERALIZED ERROR BUFFER LOADING - optimizer states compression
        if st['error_feedback']:
            lowrank_previous_momentum = previous_projector.project_back(st['m'])
            grad.add_(lowrank_previous_momentum, alpha=(beta1 / (1 - beta1)))  # store error in grad tensor
            del lowrank_previous_momentum

            grad.sub_(projector.project_back(lowdim_updated_momentum), alpha=(beta1 / (1 - beta1)))  # store error in grad tensor

        del previous_projector

        ### OPTIMIZER STATES PROJECTION-AWARE UPDATE - gradient second-order statistic
        if completed_steps > 0:
            bias1_correction = 1 - beta1**completed_steps
            bias2_correction = 1 - beta2**completed_steps

            mat_change_of_subspace.mul_(mat_change_of_subspace)

            st['v'].mul_(1 / bias2_correction)
            st['v'].addcmul_(st['m'], st['m'], value=-1 / (bias1_correction**2))

            if left_proj:
                st['v'] = torch.matmul(mat_change_of_subspace, st['v'])
            elif right_proj:
                st['v'] = torch.matmul(st['v'], mat_change_of_subspace)
            del mat_change_of_subspace

            st['v'].addcmul_(lowdim_updated_momentum, lowdim_updated_momentum, value=1 / (bias1_correction**2))
            st['v'].mul_(bias2_correction)
            st['v'].abs_()

            st['m'].copy_(lowdim_updated_momentum)
            del lowdim_updated_momentum

        ### OPTIMIZER STATES ADAM-TYPE UPDATE
        st['m'].mul_(beta1)
        st['m'].add_(lowdim_grad, alpha=(1 - beta1))

        st['v'].mul_(beta2)
        st['v'].addcmul_(lowdim_grad, lowdim_grad, value=(1 - beta2))

        ### MODEL UPDATE
        lowdim_descent_direction = st['v'].div(1 - beta2**(completed_steps + 1))
        lowdim_descent_direction.sqrt_()
        lowdim_descent_direction.add_(eps)

        lowdim_descent_direction.reciprocal_()
        lowdim_descent_direction.mul_(st['m'])
        lowdim_descent_direction.div_(1 - beta1**(completed_steps + 1))

        descent_direction = projector.project_back(lowdim_descent_direction)
        del lowdim_descent_direction

        st['previous_projector'] = projector

        p.mul_(1 - lr * wd)  # decoupled weight decay
        p.add_(descent_direction, alpha=-lr)

        del descent_direction

    def _update_lr_wd(self):
        # copy the learning rate group to parameter state because the lr scheduler updates the one in the group
        for group in self.param_groups:
            lr = group.get('lr', self.lr)  # if the param groups do not have learning rate, then use the external one
            wd = group.get('weight_decay', self.weight_decay)  # if the param groups do not have weight decay, then use the external one
            for p in group['params']:
                self.state[p]['lr'] = lr
                self.state[p]['weight_decay'] = wd

    ### GRADIENT ACCUMULATION AND ERROR BUFFER LOADING
    def zero_grad(self):
        for group in self.param_groups:
            error_feedback = group.get('error_feedback', self.error_feedback)
            if not group['enable_lowrank'] or not error_feedback:
                for p in group['params']:
                    p.grad = None
