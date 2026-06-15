# Adapted from https://github.com/konstmish/prodigy (commit 3efb213)
# Copyright (c) 2023 Konstantin Mishchenko. Licensed under the MIT license.
# See THIRD_PARTY_NOTICES.md for details.
"""Implementation of the Prodigy optimizer."""

import math

import torch
import torch.distributed as dist

from ...core.optimizer import Optimizer

__all__ = ["Prodigy"]


class Prodigy(Optimizer):
    r"""Implements Prodigy, an Adam variant that estimates its own step size online.

    .. math::
       \begin{aligned}
       m_{t+1} &= \beta_1 m_t + (1 - \beta_1)\, d_t g_t \\
       v_{t+1} &= \beta_2 v_t + (1 - \beta_2)\, d_t^2 g_t^2 \\
       r_{t+1} &= \beta_3\, r_t + \gamma_t d_t^2
                  \langle g_t, \theta_0 - \theta_t \rangle \\
       s_{t+1} &= \beta_3\, s_t + \gamma_t d_t^2 g_t \\
       \hat{d}_{t+1} &= \frac{r_{t+1}}{\|s_{t+1}\|_1}, \qquad
       d_{t+1} = \max(d_t, \hat{d}_{t+1}) \\
       \theta_{t+1} &= \theta_t - \gamma_t d_t\,
                       \frac{m_{t+1}}{\sqrt{v_{t+1}} + d_t \epsilon}
       \end{aligned}

    where :math:`d_t` estimates the distance from :math:`\theta_0` to the
    solution and :math:`\gamma_t` is the learning rate, acting only as a
    multiplier on the estimated step size. The decay rate :math:`\beta_3` of
    :math:`r_t` and :math:`s_t` defaults to :math:`\sqrt{\beta_2}` and can be
    overridden through ``beta3``. The newly added terms in :math:`r_{t+1}` and
    :math:`s_{t+1}` are accumulated without the :math:`(1 - \beta_3)`
    normalization because the constant cancels in
    :math:`\hat{d}_{t+1} = r_{t+1} / \|s_{t+1}\|_1`.

    Note:
        Leave ``lr`` at its default of 1.0. To tune the method, change
        ``d_coef``, which multiplies the estimate :math:`\hat{d}_{t+1}`.

    Reference: Konstantin Mishchenko, Aaron Defazio,
    "Prodigy: An Expeditiously Adaptive Parameter-Free Learner", ICML 2024.
    https://arxiv.org/abs/2306.06101
    """

    def __init__(self, params, lr=1.0,
                 betas=(0.9, 0.999), beta3=None,
                 eps=1e-8, weight_decay=0, decouple=True,
                 use_bias_correction=False, safeguard_warmup=False,
                 d0=1e-6, d_coef=1.0, growth_rate=float('inf'),
                 fsdp_in_use=False,
                 slice_p=1):
        if not 0.0 < d0:
            raise ValueError("Invalid d0 value: {}".format(d0))
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, beta3=beta3,
                        eps=eps, weight_decay=weight_decay,
                        d=d0, d0=d0, d_max=d0,
                        d_numerator=0.0, d_coef=d_coef,
                        k=0, growth_rate=growth_rate,
                        use_bias_correction=use_bias_correction,
                        decouple=decouple, safeguard_warmup=safeguard_warmup,
                        fsdp_in_use=fsdp_in_use,
                        slice_p=slice_p)
        self.d0 = d0
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return True

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        d_denom = 0.0

        group = self.param_groups[0]
        use_bias_correction = group['use_bias_correction']
        beta1, beta2 = group['betas']
        beta3 = group['beta3']
        if beta3 is None:
            beta3 = math.sqrt(beta2)
        k = group['k']

        d = group['d']
        d_max = group['d_max']
        d_coef = group['d_coef']
        lr = max(group['lr'] for group in self.param_groups)

        if use_bias_correction:
            bias_correction = ((1 - beta2**(k+1))**0.5) / (1 - beta1**(k+1))
        else:
            bias_correction = 1

        dlr = d*lr*bias_correction

        growth_rate = group['growth_rate']
        decouple = group['decouple']
        fsdp_in_use = group['fsdp_in_use']

        d_numerator = group['d_numerator']
        d_numerator *= beta3
        delta_numerator = 0.0

        for group in self.param_groups:
            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']
            group_lr = group['lr']
            d0 = group['d0']
            safeguard_warmup = group['safeguard_warmup']
            slice_p = group['slice_p']

            if group_lr not in [lr, 0.0]:
                raise RuntimeError("Setting different lr values in different parameter groups is only supported for values of 0")

            for p in group['params']:
                if p.grad is None:
                    continue
                if hasattr(p, "_fsdp_flattened"):
                    fsdp_in_use = True

                grad = p.grad.data

                # Apply weight decay (coupled variant)
                if decay != 0 and not decouple:
                    grad.add_(p.data, alpha=decay)

                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0

                    state['s'] = torch.zeros_like(p.data.flatten()[::slice_p]).detach()

                    if p.any():
                        state['p0'] = p.flatten()[::slice_p].detach().clone()
                    else:
                        # All values are zero, so save VRAM with a zero-tensor
                        state['p0'] = torch.tensor(0, device=p.device, dtype=p.dtype)

                    # Exponential moving average of gradient values
                    if beta1 > 0:
                        state['exp_avg'] = torch.zeros_like(p.data).detach()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data).detach()

                exp_avg_sq = state['exp_avg_sq']

                s = state['s']
                p0 = state['p0']

                if group_lr > 0.0:
                    # we use d / d0 instead of just d to avoid getting values that are too small
                    sliced_grad = grad.flatten()[::slice_p]
                    delta_numerator += (d / d0) * dlr * torch.dot(sliced_grad, p0.data - p.data.flatten()[::slice_p]).item()

                    # Adam EMA updates
                    if beta1 > 0:
                        exp_avg = state['exp_avg']
                        exp_avg.mul_(beta1).add_(grad, alpha=d * (1-beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=d * d * (1-beta2))

                    if safeguard_warmup:
                        s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * d))
                    else:
                        s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * dlr))
                    d_denom += s.abs().sum().item()

        d_hat = d

        # if we have not done any progres, return
        # if we have any gradients available, will have d_denom > 0 (unless \|g\|=0)
        if d_denom == 0 and not fsdp_in_use:
            return loss

        if lr > 0.0:
            if fsdp_in_use:
                dist_tensor = torch.zeros(2).cuda()
                dist_tensor[0] = delta_numerator
                dist_tensor[1] = d_denom
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                global_d_numerator = d_numerator + dist_tensor[0]
                global_d_denom = dist_tensor[1]
            else:
                global_d_numerator = d_numerator + delta_numerator
                global_d_denom = d_denom

            d_hat = d_coef * global_d_numerator / global_d_denom
            if d == group['d0']:
                d = max(d, d_hat)
            d_max = max(d_max, d_hat)
            d = min(d_max, d * growth_rate)

        for group in self.param_groups:
            group['d_numerator'] = global_d_numerator
            group['d_denom'] = global_d_denom
            group['d'] = d
            group['d_max'] = d_max
            group['d_hat'] = d_hat

            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                exp_avg_sq = state['exp_avg_sq']

                state['step'] += 1

                denom = exp_avg_sq.sqrt().add_(d * eps)

                # Apply weight decay (decoupled variant)
                if decay != 0 and decouple:
                    p.data.add_(p.data, alpha=-decay * dlr)

                ### Take step
                if beta1 > 0:
                    exp_avg = state['exp_avg']
                    p.data.addcdiv_(exp_avg, denom, value=-dlr)
                else:
                    p.data.addcdiv_(grad, denom, value=-dlr * d)

            group['k'] = k + 1

        return loss
