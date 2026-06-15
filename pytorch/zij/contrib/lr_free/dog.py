# Adapted from https://github.com/formll/dog (commit 8d45cf5)
# Copyright (c) 2023 Foundations of Robust Machine Learning Lab. Licensed under the MIT license.
# See THIRD_PARTY_NOTICES.md for details.
"""Implementation of the DoG and LDoG optimizers."""

import logging
from typing import Optional

import torch

from ...core.optimizer import Optimizer

__all__ = ["DoG", "LDoG"]

logger = logging.getLogger(__name__)


class DoG(Optimizer):
    r"""Implements DoG, the parameter-free distance-over-gradients step size schedule.

    .. math::
       \begin{aligned}
       \bar{r}_t &= \max\bigl(\bar{r}_{t-1},\, \lVert \theta_t - \theta_0 \rVert\bigr) \\
       G_t &= \sum_{i \le t} \lVert g_i \rVert^2 \\
       \eta_t &= \frac{\bar{r}_t}{\sqrt{G_t}} \\
       \theta_{t+1} &= \theta_t - \eta_t\, g_t
       \end{aligned}

    where the initial distance estimate is
    :math:`\bar{r}_0 = r_\epsilon = \alpha\,(1 + \lVert \theta_0 \rVert)` with
    :math:`\alpha` given by ``reps_rel``, and ``lr`` enters only as a constant
    multiplier :math:`c` on :math:`\eta_t`.

    Note:
        Leave ``lr`` at its default of 1.0. The paper recommends pairing DoG
        with polynomial decay iterate averaging, and raising ``reps_rel`` to
        1e-4 for models that use batch normalization.

    Reference: Maor Ivgi, Oliver Hinder, Yair Carmon,
    "DoG is SGD's Best Friend: A Parameter-Free Dynamic Step Size Schedule",
    ICML 2023.
    https://arxiv.org/abs/2302.12022
    """

    def __init__(self, params, reps_rel: float = 1e-6, lr: float = 1.0,
                 weight_decay: float = 0.0, eps: float = 1e-8, init_eta: Optional[float] = None):
        if lr <= 0.0:
            raise ValueError(f'Invalid learning rate ({lr}). Suggested value is 1.')
        if lr != 1.0:
            logger.warning('We do not recommend changing the lr parameter from its default value of 1')
        if init_eta is not None:
            if init_eta <= 0:
                raise ValueError(f'Invalid value for init_eta ({init_eta})')
            logger.info(f'Ignoring reps_rel since will be explicitly set init_eta to be {init_eta} (first step size)')
            reps_rel = 0
        else:
            if reps_rel <= 0.0:
                raise ValueError(f'Invalid reps_rel value ({reps_rel}). Suggested value is 1e-6 '
                                 '(unless the model uses batch-normalization, in which case suggested value is 1e-4)')

        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')

        self._first_step = True

        defaults = dict(reps_rel=reps_rel, lr=lr, weight_decay=weight_decay, eps=eps, init_eta=init_eta)
        super().__init__(params, defaults)

    def state_dict(self) -> dict:
        state_dict = super().state_dict()
        logger.info('retrieving DoG state dict')
        state_dict['state']['_first_step'] = self._first_step
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self._first_step = state_dict['state']['_first_step']
        logger.info('loaded DoG state dict')
        cuda = self.param_groups[0]['params'][0].device
        for group in self.param_groups:
            cuda_buffers = {'init_buffer'}
            for tgroup in group.keys():
                # this can cast all the tensors to the device. However, as it turns out,
                # we need ONLY the init_buffer to be on the params' device
                if tgroup != 'params':
                    device = cuda if tgroup in cuda_buffers else 'cpu'
                    if isinstance(group[tgroup], list) and len(group[tgroup]) > 0 and \
                            isinstance(group[tgroup][0], torch.Tensor):
                        group[tgroup] = [i.to(device) for i in group[tgroup]]
                    elif isinstance(group[tgroup], torch.Tensor):
                        group[tgroup] = group[tgroup].to(device)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        first_step = self._first_step

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            if first_step:
                init = group['init_buffer'] = [torch.clone(p).detach() for p in group['params']]
            else:
                init = group['init_buffer']

            if weight_decay > 0:
                for p in group['params']:
                    p.grad.add_(p, alpha=weight_decay)

            self._update_group_state(group, init)
            self._override_init_eta_if_needed(group)

            for p, eta in zip(group['params'], group['eta']):
                if p.grad is None:
                    continue
                else:
                    p.add_(p.grad, alpha=-eta)

        self._first_step = False

        return loss

    def _update_group_state(self, group, init):
        # treat all layers as one long vector
        if self._first_step:
            group['rbar'] = group['reps_rel'] * (1 + torch.stack([p.norm() for p in group['params']]).norm())
            group['G'] = torch.stack([(p.grad.detach() ** 2).sum() for p in group['params']]).sum() + group['eps']
        else:
            curr_d = torch.stack([torch.norm(p.detach() - pi) for p, pi in zip(group['params'], init)]).norm()
            group['rbar'] = torch.maximum(group['rbar'], curr_d)
            group['G'] += torch.stack([(p.grad.detach() ** 2).sum() for p in group['params']]).sum()
        assert group['G'] > 0, \
            f'DoG cannot work when G is not strictly positive. got: {group["G"]}'
        group['eta'] = [group['lr'] * group['rbar'] / torch.sqrt(group['G'])] * len(group['params'])

    def _override_init_eta_if_needed(self, group):
        # Override init_eta if needed
        if self._first_step and group['init_eta'] is not None:
            init_eta = group['init_eta']
            logger.info(f'Explicitly setting init_eta value to {init_eta}')
            group['eta'] = [eta * 0 + init_eta for eta in group['eta']]


class LDoG(DoG):
    r"""Implements LDoG, the layer-wise variant of DoG.

    Applies the DoG step size formula to each parameter tensor (layer)
    :math:`\ell` separately:

    .. math::
       \eta_t^{(\ell)} = \frac{\max_{i \le t} \lVert \theta_i^{(\ell)} -
       \theta_0^{(\ell)} \rVert}{\sqrt{\sum_{i \le t} \lVert g_i^{(\ell)}
       \rVert^2}}

    Note:
        Leave ``lr`` at its default of 1.0. The paper recommends pairing LDoG
        with polynomial decay iterate averaging.

    Reference: Maor Ivgi, Oliver Hinder, Yair Carmon,
    "DoG is SGD's Best Friend: A Parameter-Free Dynamic Step Size Schedule",
    ICML 2023.
    https://arxiv.org/abs/2302.12022
    """

    def _update_group_state(self, group, init):
        # treat each layer in the group as a separate block
        if self._first_step:
            group['rbar'] = group['reps_rel'] * (1 + torch.stack([p.norm() for p in group['params']]))
            group['G'] = torch.stack([(p.grad ** 2).sum() for p in group['params']]) + group['eps']
        else:
            curr_d = torch.stack([torch.norm(p - pi) for p, pi in zip(group['params'], init)])
            group['rbar'] = torch.maximum(group['rbar'], curr_d)
            group['G'] += torch.stack([(p.grad ** 2).sum() for p in group['params']])
        assert torch.all(group['G'] > 0).item(), \
            f'DoG cannot work when g2 is not strictly positive. got: {group["G"]}'
        group['eta'] = list(group['lr'] * group['rbar'] / torch.sqrt(group['G']))
