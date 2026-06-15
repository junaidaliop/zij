# Adapted from https://github.com/facebookresearch/dadaptation (commit c984980)
# Copyright (c) Meta Platforms, Inc. and its affiliates. Licensed under the MIT license.
# See THIRD_PARTY_NOTICES.md for details.
"""Implementation of the D-Adaptation optimizers."""

import logging
import math

import torch
import torch.distributed as dist

from ...core.optimizer import Optimizer

__all__ = ["DAdaptSGD", "DAdaptAdam"]


class DAdaptSGD(Optimizer):
    r"""Implements SGD with D-Adaptation automatic step sizes.

    .. math::
       \begin{aligned}
          \lambda_t &= \frac{d_t \gamma}{\lVert g_0 \rVert} \\
          s_{t+1} &= s_t + \lambda_t g_t \\
          z_{t+1} &= z_t - \lambda_t g_t \\
          \theta_{t+1} &= \beta \theta_t + (1 - \beta) z_{t+1} \\
          \hat{d}_{t+1} &= \frac{2 \sum_{i=0}^{t} \lambda_i
              \langle g_i, s_i \rangle}{\lVert s_{t+1} \rVert} \\
          d_{t+1} &= \max(d_t, \hat{d}_{t+1})
       \end{aligned}

    where :math:`\gamma` is ``lr`` and :math:`\beta` is ``momentum``.

    Note: ``lr`` rescales the D-adapted step size and should normally stay
    at its default of 1.0.

    Reference: Aaron Defazio and Konstantin Mishchenko,
    "Learning-Rate-Free Learning by D-Adaptation", ICML 2023.
    https://arxiv.org/abs/2301.07733
    """

    def __init__(self, params,
        lr=1.0,
        momentum=0.0,
        weight_decay=0,
        log_every=0,
        d0=1e-6, growth_rate=float('inf'),
        fsdp_in_use=False):

        if not 0.0 < d0:
            raise ValueError("Invalid d0 value: {}".format(d0))
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr,
            momentum=momentum,
            weight_decay=weight_decay, k=0,
            log_every=log_every,
            numerator_weighted=0.0,
            d=d0,
            growth_rate=growth_rate,
            fsdp_in_use=fsdp_in_use)

        try:
            self.rank = torch.distributed.get_rank()
        except Exception:
            self.rank = 0

        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        group = self.param_groups[0]
        lr = max(group['lr'] for group in self.param_groups)

        decay = group['weight_decay']
        momentum = group['momentum']
        log_every = group['log_every']
        ck = 1 - momentum
        k = group['k']

        numerator_weighted = group['numerator_weighted']
        growth_rate = group['growth_rate']
        d = group['d']
        fsdp_in_use = group['fsdp_in_use']

        group = self.param_groups[0]

        sk_sq = 0.0
        delta_numerator_weighted = 0.0

        if k == 0:
            g_sq = 0.0
            for group in self.param_groups:
                group_lr = group['lr']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    if hasattr(p, "_fsdp_flattened"):
                        fsdp_in_use = True
                    grad = p.grad.data

                    if group_lr > 0.0:
                        g_sq += (grad * grad).sum().item()

            if fsdp_in_use:
                dist_tensor = torch.zeros(1).cuda()
                dist_tensor[0] = g_sq
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                global_gsq = dist_tensor[0]
            else:
                global_gsq = g_sq
            group['g0_norm'] = g0_norm = math.sqrt(global_gsq)

        g0_norm = group['g0_norm']

        dlr = d*lr/g0_norm

        for group in self.param_groups:
            group_lr = group['lr']
            if group_lr not in [lr, 0.0]:
                raise RuntimeError("Setting different lr values in different parameter groups is only supported for values of 0")

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if 'z' not in state:
                    z = state['z'] = torch.clone(p.data).detach()
                    s = state['s'] = torch.zeros_like(p.data).detach()
                    x0 = state['x0'] = torch.clone(p.data).detach()

                # Apply weight decay
                if decay != 0:
                    grad.add_(p.data, alpha=decay)

                s = state['s']

                if group_lr > 0.0:
                    delta_numerator_weighted += dlr * torch.dot(grad.flatten(), s.flatten()).item()

                    s.data.add_(grad, alpha=dlr)
                    sk_sq += (s * s).sum().item()

        d_hat = d

        if fsdp_in_use:
            dist_tensor = torch.zeros(2).cuda()
            dist_tensor[0] = sk_sq
            dist_tensor[1] = delta_numerator_weighted
            dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
            global_sk_sq = dist_tensor[0]
            global_numerator_weighted = numerator_weighted + dist_tensor[1]
        else:
            global_sk_sq = sk_sq
            global_numerator_weighted = numerator_weighted + delta_numerator_weighted

        if lr > 0.0:
            d_hat = 2*global_numerator_weighted/math.sqrt(global_sk_sq)
            d = max(d, min(d_hat, d*growth_rate))

        # if we have not done any updates
        # if we have any gradients available, will have sk_sq > 0 (unless \|g\|=0)
        if global_sk_sq == 0:
            return loss

        if log_every > 0 and k % log_every == 0:
            logging.info(f"(r={self.rank},k={k}) dlr: {dlr} d_hat: {d_hat}, d: {d}. sk_norm={math.sqrt(global_sk_sq)} numerator_weighted={global_numerator_weighted} g0_norm={g0_norm}")

        for group in self.param_groups:
            group['numerator_weighted'] = global_numerator_weighted
            group['d'] = d
            group['g0_norm'] = g0_norm
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                s = state['s']
                x0 = state['x0']
                z = state['z']

                # z step
                z.data.copy_(x0 - s)

                # x step
                p.data.mul_(1-ck).add_(z, alpha=ck)

            group['k'] = k + 1

        return loss


class DAdaptAdam(Optimizer):
    r"""Implements Adam with D-Adaptation automatic step sizes.

    .. math::
       \begin{aligned}
          m_{t+1} &= \beta_1 m_t + (1 - \beta_1)\, d_t \gamma\, g_t \\
          v_{t+1} &= \beta_2 v_t + (1 - \beta_2)\, g_t^2 \\
          A_{t+1} &= \mathrm{diag}\bigl(\sqrt{v_{t+1}} + \epsilon\bigr) \\
          \theta_{t+1} &= \theta_t - A_{t+1}^{-1} m_{t+1} \\
          s_{t+1} &= \sqrt{\beta_2}\, s_t + (1 - \sqrt{\beta_2})\, d_t \gamma\, g_t \\
          r_{t+1} &= \sqrt{\beta_2}\, r_t + (1 - \sqrt{\beta_2})\, d_t \gamma\,
              \langle g_t, s_t \rangle_{A_t^{-1}} \\
          \hat{d}_{t+1} &= \frac{r_{t+1}}{(1 - \sqrt{\beta_2})\,
              \lVert s_{t+1} \rVert_1} \\
          d_{t+1} &= \max(d_t, \hat{d}_{t+1})
       \end{aligned}

    where :math:`\gamma` is ``lr``. Following the official implementation,
    the :math:`r` recursion weights the inner product with the pre-update
    moment matrix :math:`A_t` rather than the :math:`A_{t+1}` written in
    Algorithm 4 of the paper; the parameter step uses :math:`A_{t+1}`.

    Note: ``lr`` rescales the D-adapted step size and should normally stay
    at its default of 1.0. To scale the learning rate differently for each
    layer, set the ``layer_scale`` value of the parameter group instead.

    Reference: Aaron Defazio and Konstantin Mishchenko,
    "Learning-Rate-Free Learning by D-Adaptation", ICML 2023.
    https://arxiv.org/abs/2301.07733
    """

    def __init__(self, params, lr=1.0,
                 betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, log_every=0,
                 decouple=False,
                 use_bias_correction=False,
                 d0=1e-6, growth_rate=float('inf'),
                 fsdp_in_use=False):
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

        if decouple:
            print("Using decoupled weight decay")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        d=d0,
                        k=0,
                        layer_scale=1.0,
                        numerator_weighted=0.0,
                        log_every=log_every,
                        growth_rate=growth_rate,
                        use_bias_correction=use_bias_correction,
                        decouple=decouple,
                        fsdp_in_use=fsdp_in_use)
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

        sk_l1 = 0.0

        group = self.param_groups[0]
        use_bias_correction = group['use_bias_correction']
        numerator_weighted = group['numerator_weighted']
        beta1, beta2 = group['betas']
        k = group['k']

        d = group['d']
        lr = max(group['lr'] for group in self.param_groups)

        if use_bias_correction:
            bias_correction = ((1-beta2**(k+1))**0.5)/(1-beta1**(k+1))
        else:
            bias_correction = 1

        dlr = d*lr*bias_correction

        growth_rate = group['growth_rate']
        decouple = group['decouple']
        log_every = group['log_every']
        fsdp_in_use = group['fsdp_in_use']

        sqrt_beta2 = beta2**(0.5)

        numerator_acum = 0.0

        for group in self.param_groups:
            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']
            group_lr = group['lr']
            r = group['layer_scale']

            if group_lr not in [lr, 0.0]:
                raise RuntimeError("Setting different lr values in different parameter groups "
                                   "is only supported for values of 0. To scale the learning "
                                   "rate differently for each layer, set the 'layer_scale' value instead.")

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
                    state['s'] = torch.zeros_like(p.data).detach()
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data).detach()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data).detach()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                s = state['s']

                if group_lr > 0.0:
                    denom = exp_avg_sq.sqrt().add_(eps)
                    numerator_acum += r * dlr * torch.dot(grad.flatten(), s.div(denom).flatten()).item()

                    # Adam EMA updates
                    exp_avg.mul_(beta1).add_(grad, alpha=r*dlr*(1-beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                    s.mul_(sqrt_beta2).add_(grad, alpha=dlr*(1-sqrt_beta2))
                    sk_l1 += r * s.abs().sum().item()

        d_hat = d

        # if we have not done any progres, return
        # if we have any gradients available, will have sk_l1 > 0 (unless \|g\|=0)
        if sk_l1 == 0:
            return loss

        if fsdp_in_use:
            dist_tensor = torch.zeros(2).cuda()
            dist_tensor[0] = numerator_acum
            dist_tensor[1] = sk_l1
            dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
            global_numerator_weighted = sqrt_beta2*numerator_weighted + (1-sqrt_beta2)*dist_tensor[0]
            global_sk_l1 = dist_tensor[1]
        else:
            global_numerator_weighted = sqrt_beta2*numerator_weighted + (1-sqrt_beta2)*numerator_acum
            global_sk_l1 = sk_l1

        if lr > 0.0:
            d_hat = global_numerator_weighted/((1-sqrt_beta2)*global_sk_l1)
            d = max(d, min(d_hat, d*growth_rate))

        if log_every > 0 and k % log_every == 0:
            logging.info(f"lr: {lr} dlr: {dlr} d_hat: {d_hat}, d: {d}. sk_l1={global_sk_l1:1.1e} numerator_weighted={global_numerator_weighted:1.1e}")

        for group in self.param_groups:
            group['numerator_weighted'] = global_numerator_weighted
            group['d'] = d

            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1

                denom = exp_avg_sq.sqrt().add_(eps)

                # Apply weight decay (decoupled variant)
                if decay != 0 and decouple:
                    p.data.add_(p.data, alpha=-decay * dlr)

                ### Take step
                p.data.addcdiv_(exp_avg, denom, value=-1)

            group['k'] = k + 1

        return loss
