# Adapted from https://github.com/fabian-sp/MoMo (commit 7be52f2)
# Copyright (c) 2023 fabian-sp. Licensed under the MIT license.
# See THIRD_PARTY_NOTICES.md for details.
"""Implementation of the MoMo optimizers."""

import warnings

import torch

from ...core.optimizer import Optimizer

__all__ = ["Momo", "MomoAdam"]


class Momo(Optimizer):
    r"""Implements MoMo, SGD with momentum and an adaptive Polyak step size.

    .. math::
       \begin{aligned}
       \bar{f}_t &= \beta \bar{f}_{t-1} + (1 - \beta) f_t \\
       \gamma_t &= \beta \gamma_{t-1} + (1 - \beta) \langle g_t, \theta_t \rangle \\
       m_t &= \beta m_{t-1} + (1 - \beta) g_t \\
       h_t &= \bar{f}_t + \langle m_t, \theta_t \rangle - \gamma_t \\
       \theta_{t+1} &= \theta_t - \min\left\{ \eta,
           \frac{(h_t - f_*)_+}{\lVert m_t \rVert^2} \right\} m_t
       \end{aligned}

    where :math:`f_t` is the loss, :math:`f_*` is the lower bound ``lb`` on
    the loss, and ``lr`` sets the cap :math:`\eta` on the adaptive step size.
    With ``bias_correction=True`` the averages start at zero and :math:`f_*`
    and :math:`\eta` are rescaled by :math:`\rho_t = 1 - \beta^t`; with
    ``weight_decay`` :math:`\lambda > 0` the update ends with a proximal
    division by :math:`1 + \eta\lambda`. ``use_fstar=True`` estimates the
    lower bound online instead of keeping it fixed.

    Note:
        ``step`` needs the current loss value: pass either a closure or, if
        the backward pass already ran, the loss tensor through ``loss``.

    Reference: Fabian Schaipp, Ruben Ohana, Michael Eickenberg,
    Aaron Defazio, Robert M. Gower,
    "MoMo: Momentum Models for Adaptive Learning Rates", ICML 2024.
    https://arxiv.org/abs/2305.07583
    """

    def __init__(self, params, lr=1.0, weight_decay=0, beta=0.9, lb=0,
                 bias_correction=False, use_fstar=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        if (beta < 0.0) or (beta > 1.0):
            raise ValueError("Invalid beta parameter: {}".format(beta))

        defaults = dict(lr=lr, weight_decay=weight_decay)

        super().__init__(params, defaults)

        self.beta = beta
        self.lb = lb
        self._initial_lb = lb
        self.bias_correction = bias_correction
        self.use_fstar = use_fstar

        self._number_steps = 0
        self.state['step_size_list'] = list()  # for storing the adaptive step size term

    def step(self, closure=None, loss=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            loss (Tensor, optional): The loss tensor. Use this when the
                backward pass has already been performed.
        """
        assert (closure is not None) or (loss is not None), "Either loss tensor or closure must be passed."
        assert (closure is None) or (loss is None), "Pass either the loss tensor or the closure, not both."

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if len(self.param_groups) > 1:
            warnings.warn("More than one param group. step_size_list contains adaptive term of last group.")
            warnings.warn("More than one param group. This might cause issues for the step method.")

        self._number_steps += 1
        beta = self.beta

        if self._number_steps == 1:
            if self.bias_correction:
                self.loss_avg = 0.
            else:
                self.loss_avg = loss.detach().clone()

        self.loss_avg = beta * self.loss_avg + (1 - beta) * loss.detach()

        if self.bias_correction:
            rho = 1 - beta ** self._number_steps  # must be after incrementing k
        else:
            rho = 1

        _dot = 0.
        _gamma = 0.
        _norm = 0.

        # Notation
        # d_k: p.grad_avg, gamma_k: _gamma, \bar f_k: self.loss_avg
        for group in self.param_groups:
            for p in group['params']:

                grad = p.grad.data.detach()
                state = self.state[p]

                # Initialize EMA
                if self._number_steps == 1:
                    if self.bias_correction:
                        state['grad_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()
                        state['grad_dot_w'] = torch.zeros(1).to(p.device)
                    else:
                        # Exponential moving average of gradients
                        state['grad_avg'] = grad.clone()
                        # Exponential moving average of inner product <grad, weight>
                        state['grad_dot_w'] = torch.sum(torch.mul(p.data, grad))

                grad_avg, grad_dot_w = state['grad_avg'], state['grad_dot_w']

                grad_avg.mul_(beta).add_(grad, alpha=1 - beta)
                grad_dot_w.mul_(beta).add_(torch.sum(torch.mul(p.data, grad)), alpha=1 - beta)

                _dot += torch.sum(torch.mul(p.data, grad_avg))
                _gamma += grad_dot_w
                _norm += torch.sum(torch.mul(grad_avg, grad_avg))

        # Update
        for group in self.param_groups:
            lr = group['lr']
            lmbda = group['weight_decay']

            if self.use_fstar:
                cap = ((1 + lr * lmbda) * self.loss_avg + _dot - (1 + lr * lmbda) * _gamma).item()
                # Reset
                if cap < (1 + lr * lmbda) * rho * self.lb:
                    self.lb = cap / (2 * (1 + lr * lmbda) * rho)
                    self.lb = max(self.lb, self._initial_lb)  # safeguard

            # Compute adaptive step size
            if lmbda > 0:
                nom = (1 + lr * lmbda) * (self.loss_avg - rho * self.lb) + _dot - (1 + lr * lmbda) * _gamma
                t1 = max(nom, 0.) / _norm
            else:
                t1 = max(self.loss_avg - rho * self.lb + _dot - _gamma, 0.) / _norm

            t1 = t1.item()  # make scalar

            tau = min(lr / rho, t1)  # step size

            # Update lb estimator
            if self.use_fstar:
                h = (self.loss_avg + _dot - _gamma).item()
                self.lb = ((h - (1 / 2) * tau * _norm) / rho).item()
                self.lb = max(self.lb, self._initial_lb)  # safeguard

            # Update params
            for p in group['params']:
                state = self.state[p]
                grad_avg = state['grad_avg']
                p.data.add_(other=grad_avg, alpha=-tau)

                if lmbda > 0:
                    p.data.div_(1 + lr * lmbda)

        if self.use_fstar:
            self.state['fstar'] = self.lb

        return loss


class MomoAdam(Optimizer):
    r"""Implements MoMo-Adam, Adam with an adaptive Polyak step size.

    .. math::
       \begin{aligned}
       m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
       v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, g_t \odot g_t \\
       D_t &= \mathrm{Diag}\left(\sqrt{v_t / (1 - \beta_2^t)} + \epsilon\right) \\
       \bar{f}_t &= \beta_1 \bar{f}_{t-1} + (1 - \beta_1) f_t \\
       \gamma_t &= \beta_1 \gamma_{t-1} + (1 - \beta_1) \langle g_t, \theta_t \rangle \\
       \tau_t &= \min\left\{ \frac{\eta}{1 - \beta_1^t},
           \frac{\left((1 + \eta\lambda)\left(\bar{f}_t - \gamma_t
           - (1 - \beta_1^t) f_*\right)
           + \langle m_t, \theta_t \rangle\right)_+}
           {\lVert m_t \rVert^2_{D_t^{-1}}} \right\} \\
       \theta_{t+1} &= \frac{1}{1 + \eta\lambda}
           \left(\theta_t - \tau_t D_t^{-1} m_t\right)
       \end{aligned}

    where :math:`f_t` is the loss, :math:`f_*` is the lower bound ``lb`` on
    the loss, :math:`\eta` is ``lr``, and :math:`\lambda` is ``weight_decay``.
    ``divide=False`` replaces the proximal division by the AdamW-style decay
    :math:`\theta_t \leftarrow (1 - \eta\lambda)\,\theta_t` before the step;
    ``use_fstar=True`` estimates the lower bound online.

    Note:
        ``step`` needs the current loss value: pass either a closure or, if
        the backward pass already ran, the loss tensor through ``loss``.

    Reference: Fabian Schaipp, Ruben Ohana, Michael Eickenberg,
    Aaron Defazio, Robert M. Gower,
    "MoMo: Momentum Models for Adaptive Learning Rates", ICML 2024.
    https://arxiv.org/abs/2305.07583
    """

    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, lb=0, divide=True, use_fstar=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)

        super().__init__(params, defaults)

        self.lb = lb
        self._initial_lb = lb
        self.divide = divide
        self.use_fstar = use_fstar

        self._number_steps = 0
        self.loss_avg = 0.
        self.state['step_size_list'] = list()  # for storing the adaptive step size term

    def step(self, closure=None, loss=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            loss (Tensor, optional): The loss tensor. Use this when the
                backward pass has already been performed.
        """
        assert (closure is not None) or (loss is not None), "Either loss tensor or closure must be passed."
        assert (closure is None) or (loss is None), "Pass either the loss tensor or the closure, not both."

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if len(self.param_groups) > 1:
            warnings.warn("More than one param group. step_size_list contains adaptive term of last group.")
            warnings.warn("More than one param group. This might cause issues for the step method.")

        _dot = 0.  # = <d_k,x_k>
        _gamma = 0.  # = gamma_k
        _grad_norm = 0.  # = ||d_k||^2_{D_k^-1}

        self._number_steps += 1

        for group in self.param_groups:
            eps = group['eps']
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                    # Exponential moving average of gradients
                    state['grad_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()
                    # Exponential moving average of squared gradient values
                    state['grad_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format).detach()
                    # Exponential moving average of inner product <grad, weight>
                    state['grad_dot_w'] = torch.tensor(0.).to(p.device)

                state['step'] += 1
                grad_avg, grad_avg_sq = state['grad_avg'], state['grad_avg_sq']
                grad_dot_w = state['grad_dot_w']

                # Adam EMA updates
                grad_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # = d_k
                grad_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # = v_k
                grad_dot_w.mul_(beta1).add_(torch.sum(torch.mul(p.data, grad)), alpha=1 - beta1)

                bias_correction2 = 1 - beta2 ** state['step']
                Dk = grad_avg_sq.div(bias_correction2).sqrt().add(eps)  # = D_k

                _dot += torch.sum(torch.mul(p.data, grad_avg))
                _gamma += grad_dot_w
                _grad_norm += torch.sum(grad_avg.mul(grad_avg.div(Dk)))

        # Exponential moving average of function value
        # Uses beta1 of last param_group!
        self.loss_avg = (1 - beta1) * loss.detach() + beta1 * self.loss_avg

        # Update
        for group in self.param_groups:

            # Compute adaptive step size
            lr = group['lr']
            lmbda = group['weight_decay']
            eps = group['eps']
            beta1, beta2 = group['betas']

            bias_correction1 = 1 - beta1 ** self._number_steps
            bias_correction2 = 1 - beta2 ** self._number_steps

            if self.use_fstar:
                cap = ((1 + lr * lmbda) * self.loss_avg + _dot - (1 + lr * lmbda) * _gamma).item()
                # Reset
                if cap < (1 + lr * lmbda) * bias_correction1 * self.lb:
                    self.lb = cap / (2 * (1 + lr * lmbda) * bias_correction1)
                    self.lb = max(self.lb, self._initial_lb)  # safeguard

            nom = (1 + lr * lmbda) * (self.loss_avg - bias_correction1 * self.lb) + _dot - (1 + lr * lmbda) * _gamma

            t1 = (max(nom, 0.) / _grad_norm).item()
            tau = min(lr / bias_correction1, t1)

            # Update lb estimator
            if self.use_fstar:
                h = (self.loss_avg + _dot - _gamma).item()
                self.lb = ((h - (1 / 2) * tau * _grad_norm) / bias_correction1).item()
                self.lb = max(self.lb, self._initial_lb)  # safeguard

            # Update params
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                grad_avg, grad_avg_sq = state['grad_avg'], state['grad_avg_sq']

                Dk = grad_avg_sq.div(bias_correction2).sqrt().add(eps)

                # AdamW-Pytorch way of weight decay
                if lmbda > 0 and not self.divide:
                    p.data.mul_(1 - lmbda * lr)

                # Gradient step
                p.data.addcdiv_(grad_avg, Dk, value=-tau)  # x_k - tau*(d_k/D_k)

                # Proximal way of weight decay
                if lmbda > 0 and self.divide:
                    p.data.div_(1 + lmbda * lr)

        if self.use_fstar:
            self.state['fstar'] = self.lb

        return loss
