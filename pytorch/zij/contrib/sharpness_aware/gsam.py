# Adapted from https://github.com/juntang-zhuang/GSAM (commit a770275)
# Copyright (c) 2021 David Samuel. Licensed under the MIT license.
# See THIRD_PARTY_NOTICES.md for details.
# mypy: allow-untyped-defs
r"""Implementation of the GSAM optimizer."""

import contextlib

import torch
from torch.distributed import ReduceOp
from torch.nn.modules.batchnorm import _BatchNorm

from ...core import SGD
from ...core.optimizer import Optimizer

__all__ = ["GSAM", "ProportionScheduler"]


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


class ProportionScheduler:
    """Outputs a value that evolves proportionally to the learning rate of a
    PyTorch scheduler, so that
    (value - min_value) / (max_value - min_value) = (lr - min_lr) / (max_lr - min_lr).
    """

    def __init__(self, pytorch_lr_scheduler, max_lr, min_lr, max_value, min_value):
        self.t = 0
        self.pytorch_lr_scheduler = pytorch_lr_scheduler
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.max_value = max_value
        self.min_value = min_value

        assert (max_lr > min_lr) or ((max_lr == min_lr) and (max_value == min_value)), (
            "`value` is scheduled to evolve proportionally to `lr`, e.g. "
            "`(lr - min_lr) / (max_lr - min_lr) = (value - min_value) / (max_value - min_value)`. "
            "Check `max_lr >= min_lr` and `max_value >= min_value`; if `max_lr == min_lr`, "
            "set `max_value == min_value` so `value` is constant with step."
        )
        assert max_value >= min_value

        self.step()  # take 1 step during initialization to get self._last_lr

    def lr(self):
        return self._last_lr[0]

    def step(self):
        self.t += 1
        if hasattr(self.pytorch_lr_scheduler, "_last_lr"):
            lr = self.pytorch_lr_scheduler._last_lr[0]
        else:
            lr = self.pytorch_lr_scheduler.optimizer.param_groups[0]["lr"]

        if self.max_lr > self.min_lr:
            value = self.min_value + (self.max_value - self.min_value) * (
                lr - self.min_lr
            ) / (self.max_lr - self.min_lr)
        else:
            value = self.max_value

        self._last_lr = [value]
        return value


class GSAM(Optimizer):
    r"""Implements GSAM, surrogate gap guided sharpness-aware minimization
    wrapping a base optimizer.

    .. math::
       \begin{aligned}
       &\theta_t^{adv} = \theta_t
           + \rho_t \frac{g_t}{\lVert g_t \rVert + \epsilon}, \qquad
        g_t^p = \nabla f(\theta) \big\rvert_{\theta = \theta_t^{adv}}      \\
       &g_t = g_{t,\parallel} + g_{t,\perp}, \qquad
        g_{t,\parallel} = \frac{\langle g_t, g_t^p \rangle}
           {\lVert g_t^p \rVert^2} \, g_t^p                                \\
       &\theta_{t+1} = \theta_t - \eta_t \left( g_t^p
           - \alpha \, g_{t,\perp} \right)
       \end{aligned}

    where :math:`g_t` is the gradient of the loss :math:`f`,
    :math:`g_t^p` is the gradient of the perturbed loss, and the
    perturbation radius follows the schedule
    :math:`\rho_t = \rho_{min} + (\rho_{max} - \rho_{min})
    (\eta_t - \eta_{min}) / (\eta_{max} - \eta_{min})`.

    Reference: Juntang Zhuang et al., "Surrogate Gap Minimization Improves
    Sharpness-Aware Training", ICLR 2022.
    https://arxiv.org/abs/2203.08065

    Note:
        Each step needs two forward-backward passes: call
        :meth:`set_closure` and then :meth:`step`, or pass ``step`` a closure
        that zeroes gradients, computes the loss, and calls ``backward()``.
        Pass ``rho_scheduler`` (for example :class:`ProportionScheduler`) and
        call :meth:`update_rho_t` once per step to anneal :math:`\rho_t` with
        the learning rate as in the paper; without one, :math:`\rho_t` stays
        at the constant ``rho``. The state dict covers only the GSAM
        perturbation state; save ``base_optimizer.state_dict()`` separately.
    """

    def __init__(
        self,
        params,
        base_optimizer=None,
        model=None,
        gsam_alpha=0.1,
        rho_scheduler=None,
        adaptive=False,
        perturb_eps=1e-12,
        grad_reduce="mean",
        rho=0.05,
        **kwargs,
    ):
        defaults = dict(adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        if base_optimizer is None:
            base_optimizer = SGD
        if isinstance(base_optimizer, type):
            base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive = adaptive
        self.rho_scheduler = rho_scheduler
        self.rho = rho
        self.perturb_eps = perturb_eps
        self.alpha = gsam_alpha

        # initialize self.rho_t
        self.update_rho_t()

        # set up reduction for gradient across workers
        if grad_reduce.lower() == "mean":
            if hasattr(ReduceOp, "AVG"):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else:  # PyTorch <= 1.11.0 does not have AVG, manually average across processes
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == "sum":
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')

    @torch.no_grad()
    def update_rho_t(self):
        if self.rho_scheduler is not None:
            self.rho_t = self.rho_scheduler.step()
        else:
            self.rho_t = self.rho
        return self.rho_t

    @torch.no_grad()
    def perturb_weights(self, rho=0.0):
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        for group in self.param_groups:
            scale = rho / (grad_norm + self.perturb_eps)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_g"] = p.grad.data.clone()
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for p in group["params"]:
                if "e_w" in self.state[p].keys():
                    p.data.sub_(self.state[p]["e_w"])

    @torch.no_grad()
    def gradient_decompose(self, alpha=0.0):
        # calculate inner product
        inner_prod = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                inner_prod += torch.sum(self.state[p]["old_g"] * p.grad.data)

        # get norm
        new_grad_norm = self._grad_norm()
        old_grad_norm = self._grad_norm(by="old_g")

        # get cosine
        cosine = inner_prod / (new_grad_norm * old_grad_norm + self.perturb_eps)

        # gradient decomposition
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                vertical = self.state[p]["old_g"] - cosine * old_grad_norm * p.grad.data / (
                    new_grad_norm + self.perturb_eps
                )
                p.grad.data.add_(vertical, alpha=-alpha)

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized():  # synchronize final gradients
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    @torch.no_grad()
    def _grad_norm(self, by=None, weight_adaptive=False):
        if not by:
            norm = torch.norm(
                torch.stack(
                    [
                        ((torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2)
                        for group in self.param_groups
                        for p in group["params"]
                        if p.grad is not None
                    ]
                ),
                p=2,
            )
        else:
            norm = torch.norm(
                torch.stack(
                    [
                        ((torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                        for group in self.param_groups
                        for p in group["params"]
                        if p.grad is not None
                    ]
                ),
                p=2,
            )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def maybe_no_sync(self):
        if torch.distributed.is_initialized() and hasattr(self.model, "no_sync"):
            return self.model.no_sync()
        return contextlib.ExitStack()

    @torch.no_grad()
    def set_closure(self, loss_fn, inputs, targets, **kwargs):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.
        # This function does not take any arguments, and the inputs and targets data
        # should be pre-set in the definition of partial-function

        def get_grad():
            self.base_optimizer.zero_grad()
            with torch.enable_grad():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets, **kwargs)
            loss_value = loss.data.clone().detach()
            loss.backward()
            return outputs, loss_value

        self.forward_backward_func = get_grad

    @torch.no_grad()
    def step(self, closure=None):
        if closure:
            get_grad = torch.enable_grad()(closure)
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            # get gradient
            result = get_grad()
            outputs, loss_value = result if isinstance(result, tuple) else (None, result)

            # perturb weights
            self.perturb_weights(rho=self.rho_t)

            # disable running stats for second pass
            if self.model is not None:
                disable_running_stats(self.model)

            # get gradient at perturbed weights
            get_grad()

            # decompose and get new update direction
            self.gradient_decompose(self.alpha)

            # unperturb
            self.unperturb()

        # synchronize gradients across workers
        self._sync_grad()

        # update with new directions
        self.base_optimizer.step()

        # enable running stats
        if self.model is not None:
            enable_running_stats(self.model)

        return outputs, loss_value
