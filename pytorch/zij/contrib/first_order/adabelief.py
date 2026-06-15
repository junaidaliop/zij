# Adapted from https://github.com/juntang-zhuang/Adabelief-Optimizer (commit 2855178)
# Copyright (c) 2021 Juntang Zhuang. Licensed under the BSD-2-Clause license.
# See THIRD_PARTY_NOTICES.md for details.
"""Implementation of the AdaBelief optimizer."""

import math

import torch

from ...core.optimizer import Optimizer

__all__ = ["AdaBelief"]


class AdaBelief(Optimizer):
    r"""Implements AdaBelief, an Adam variant that scales the step size by the
    belief in the observed gradient.

    AdaBelief replaces Adam's second moment :math:`v_t` (the running average of
    :math:`g_t^2`) with :math:`s_t`, the running average of the squared deviation
    of the gradient from its own first moment :math:`(g_t - m_t)^2`. A small
    deviation signals a trustworthy gradient direction and yields a large step;
    a large deviation yields a small step.

    .. math::
       \begin{aligned}
       m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
       s_t &= \beta_2 s_{t-1} + (1 - \beta_2) (g_t - m_t)^2 \\
       \hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \qquad
       \hat{s}_t = \frac{s_t}{1 - \beta_2^t} \\
       \theta_t &= \theta_{t-1} - \frac{\eta\, \hat{m}_t}{\sqrt{\hat{s}_t} + \epsilon}
       \end{aligned}

    where :math:`\theta` are the parameters, :math:`\eta` is the learning rate,
    :math:`g_t` is the gradient, :math:`m_t` and :math:`s_t` are the first moment
    and the belief in the gradient, and :math:`\beta_1, \beta_2` are the decay
    rates of the moving averages.

    The equations above describe the ``rectify=False`` path. The default
    ``rectify=True`` instead applies the RAdam variance rectification: when the
    length of the approximated moving average is large enough the step is
    rescaled by the RAdam factor and the denominator uses the un-bias-corrected
    :math:`\sqrt{s_t}`, otherwise it reduces to an SGD-like step on
    :math:`\hat{m}_t`. Following the official implementation, :math:`\epsilon`
    is added to :math:`s_t` before the square root and again after it.

    Reference: Juntang Zhuang, Tommy Tang, Yifan Ding, Sekhar Tatikonda,
    Nicha Dvornek, Xenophon Papademetris, James S. Duncan, "AdaBelief Optimizer:
    Adapting Stepsizes by the Belief in Observed Gradients", NeurIPS 2020.
    https://arxiv.org/abs/2010.07468
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16,
                 weight_decay=0, amsgrad=False, weight_decouple=True,
                 fixed_decay=False, rectify=True, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        buffer=[[None, None, None] for _ in range(10)])
        super().__init__(params, defaults)

        self.weight_decouple = weight_decouple
        self.rectify = rectify
        self.fixed_decay = fixed_decay

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                amsgrad = group['amsgrad']

                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                # Exponential moving average of the squared gradient deviation
                state['exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                if amsgrad:
                    # Maintains the maximum of all squared deviation running averages
                    state['max_exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # cast data type
                half_precision = False
                if p.data.dtype == torch.float16:
                    half_precision = True
                    p.data = p.data.float()
                    p.grad = p.grad.float()

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'AdaBelief does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                beta1, beta2 = group['betas']

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of the squared gradient deviation
                    state['exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains the maximum of all squared deviation running averages
                        state['max_exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                # perform weight decay, check if decoupled weight decay
                if self.weight_decouple:
                    if not self.fixed_decay:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        p.data.mul_(1.0 - group['weight_decay'])
                else:
                    if group['weight_decay'] != 0:
                        grad.add_(p.data, alpha=group['weight_decay'])

                # get current state variable
                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Update first and second moment running average
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

                if amsgrad:
                    max_exp_avg_var = state['max_exp_avg_var']
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_var, exp_avg_var.add_(group['eps']), out=max_exp_avg_var)

                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # update
                if not self.rectify:
                    # Default update
                    step_size = group['lr'] / bias_correction1
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

                else:  # Rectified update, forked from RAdam
                    buffered = group['buffer'][int(state['step'] % 10)]
                    if state['step'] == buffered[0]:
                        N_sma, step_size = buffered[1], buffered[2]
                    else:
                        buffered[0] = state['step']
                        beta2_t = beta2 ** state['step']
                        N_sma_max = 2 / (1 - beta2) - 1
                        N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                        buffered[1] = N_sma

                        # more conservative since it's an approximated value
                        if N_sma >= 5:
                            step_size = math.sqrt(
                                (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                        elif self.degenerated_to_sgd:
                            step_size = 1.0 / (1 - beta1 ** state['step'])
                        else:
                            step_size = -1
                        buffered[2] = step_size

                    if N_sma >= 5:
                        denom = exp_avg_var.sqrt().add_(group['eps'])
                        p.data.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                    elif step_size > 0:
                        p.data.add_(exp_avg, alpha=-step_size * group['lr'])

                if half_precision:
                    p.data = p.data.half()
                    p.grad = p.grad.half()

        return loss
