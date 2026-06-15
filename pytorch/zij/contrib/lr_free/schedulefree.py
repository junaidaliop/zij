# Adapted from https://github.com/facebookresearch/schedule_free (commit 43e5c2d)
# Copyright (c) Meta Platforms, Inc. and affiliates. Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
"""Schedule-Free optimizers: SGD, AdamW, and RAdam variants."""

from typing import Callable, Optional, Tuple, Union

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["SGDScheduleFree", "AdamWScheduleFree", "RAdamScheduleFree"]


class SGDScheduleFree(Optimizer):
    r"""Implements Schedule-Free SGD, which replaces momentum with interpolation and averaging.

    .. math::
       \begin{aligned}
           y_t &= (1 - \beta) z_t + \beta x_t, \\
           z_{t+1} &= z_t - \gamma_t \bigl(\nabla f(y_t) + \lambda y_t\bigr), \\
           x_{t+1} &= (1 - c_{t+1}) x_t + c_{t+1} z_{t+1},
       \end{aligned}

    where :math:`z_t` is the base SGD iterate, gradients are evaluated at the
    interpolated point :math:`y_t`, the parameters used for evaluation are the
    average :math:`x_t`, :math:`\lambda` is ``weight_decay``, and
    :math:`c_{t+1} = \gamma_t^2 / \sum_{i=1}^{t} \gamma_i^2`. No learning
    rate schedule is needed; linear warmup is available through
    ``warmup_steps``.

    Reference: Aaron Defazio, Xingyu Yang, Harsh Mehta, Konstantin Mishchenko,
    Ahmed Khaled, Ashok Cutkosky, "The Road Less Scheduled", NeurIPS 2024.
    https://arxiv.org/abs/2405.15682

    Note:
        Call ``optimizer.train()`` before training and ``optimizer.eval()``
        before evaluation or checkpointing, alongside the matching
        ``model.train()`` / ``model.eval()`` calls. Gradients are computed at
        :math:`y_t` while losses should be measured at :math:`x_t`, so the
        parameter buffer must be switched between the two points.
    """

    def __init__(self,
                 params: ParamsT,
                 lr: Union[float, torch.Tensor] = 1.0,
                 momentum: float = 0.9,
                 weight_decay: float = 0,
                 warmup_steps: int = 0,
                 r: float = 0.0,
                 weight_lr_power: float = 2,
                 foreach: Optional[bool] = hasattr(torch, "_foreach_mul_"),
                 ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if momentum <= 0 or momentum >= 1:
            raise ValueError("Momentum must be between 0 and 1 exclusive: {}".format(momentum))

        defaults = dict(lr=lr,
                        momentum=momentum,
                        r=r,
                        k=0,
                        warmup_steps=warmup_steps,
                        train_mode=False,
                        weight_sum=0.0,
                        lr_max=-1.0,
                        scheduled_lr=0.0,
                        weight_lr_power=weight_lr_power,
                        weight_decay=weight_decay,
                        foreach=foreach)
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            momentum = group['momentum']
            if train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p to x
                        p.lerp_(end=state['z'].to(p.device), weight=1 - 1 / momentum)
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            momentum = group['momentum']
            if not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p to y
                        p.lerp_(end=state['z'].to(p.device), weight=1 - momentum)
                group['train_mode'] = True

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if not self.param_groups[0]['train_mode']:
            raise Exception("Optimizer was not in train mode when step is called. "
                            "Please insert .train() and .eval() calls on the "
                            "optimizer. See documentation for details.")

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            lr = group['lr']
            weight_decay = group['weight_decay']
            k = group['k']
            warmup_steps = group['warmup_steps']

            if k < warmup_steps:
                sched = (k + 1) / warmup_steps
            else:
                sched = 1.0
            lr = group['lr'] * sched
            group['scheduled_lr'] = lr  # For logging purposes

            weight_lr_power = group['weight_lr_power']

            r = group['r']
            lr_max = group['lr_max'] = max(lr, group['lr_max'])

            weight = ((k + 1) ** r) * (lr_max ** weight_lr_power)
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            try:
                ckp1 = weight / weight_sum
            except ZeroDivisionError:
                ckp1 = 0

            active_p = [p for p in group['params'] if p.grad is not None]

            for p in active_p:
                if 'z' not in self.state[p]:
                    self.state[p]['z'] = torch.clone(p, memory_format=torch.preserve_format)

            if group['foreach'] and len(active_p) > 0:
                y, grad, z = zip(*[(p, p.grad, self.state[p]['z'])
                                   for p in active_p])

                # Apply weight decay
                if weight_decay != 0:
                    torch._foreach_add_(grad, y, alpha=weight_decay)

                # These operations update y in-place,
                # without computing x explicitly.
                torch._foreach_lerp_(y, z, weight=ckp1)
                torch._foreach_add_(y, grad, alpha=lr * (momentum * (1 - ckp1) - 1))

                # SGD step
                torch._foreach_sub_(z, grad, alpha=lr)
            else:
                for p in active_p:
                    y = p  # Notation to match theory
                    grad = p.grad
                    z = self.state[p]['z']

                    # Apply weight decay
                    if weight_decay != 0:
                        grad.add_(y, alpha=weight_decay)

                    # These operations update y in-place,
                    # without computing x explicitly.
                    y.lerp_(end=z, weight=ckp1)
                    y.add_(grad, alpha=lr * (momentum * (1 - ckp1) - 1))

                    # SGD step
                    z.sub_(grad, alpha=lr)

            group['k'] = k + 1
        return loss


class AdamWScheduleFree(Optimizer):
    r"""Implements Schedule-Free AdamW, AdamW with momentum replaced by interpolation and averaging.

    .. math::
       \begin{aligned}
           y_t &= (1 - \beta_1) z_t + \beta_1 x_t, \\
           v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2, \qquad
               \hat{v}_t = v_t / (1 - \beta_2^t), \\
           z_{t+1} &= z_t - \gamma_t \left( \frac{g_t}{\sqrt{\hat{v}_t}
               + \epsilon} + \lambda y_t \right), \\
           x_{t+1} &= (1 - c_{t+1}) x_t + c_{t+1} z_{t+1},
       \end{aligned}

    where gradients :math:`g_t` are evaluated at the interpolated point
    :math:`y_t`, the parameters used for evaluation are the average
    :math:`x_t`, and :math:`c_{t+1} = \gamma_t^2 / \sum_{i=1}^{t}
    \gamma_i^2`. No learning rate schedule is needed; linear warmup is
    available through ``warmup_steps``.

    Reference: Aaron Defazio, Xingyu Yang, Harsh Mehta, Konstantin Mishchenko,
    Ahmed Khaled, Ashok Cutkosky, "The Road Less Scheduled", NeurIPS 2024.
    https://arxiv.org/abs/2405.15682

    Note:
        Call ``optimizer.train()`` before training and ``optimizer.eval()``
        before evaluation or checkpointing, alongside the matching
        ``model.train()`` / ``model.eval()`` calls. Gradients are computed at
        :math:`y_t` while losses should be measured at :math:`x_t`, so the
        parameter buffer must be switched between the two points.
    """

    def __init__(self,
                 params: ParamsT,
                 lr: Union[float, torch.Tensor] = 0.0025,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 warmup_steps: int = 0,
                 r: float = 0.0,
                 weight_lr_power: float = 2.0,
                 inner_momentum: float = 0.0,
                 foreach: Optional[bool] = hasattr(torch, "_foreach_mul_")
                 ):

        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        r=r,
                        k=0,
                        warmup_steps=warmup_steps,
                        train_mode=False,
                        weight_sum=0.0,
                        lr_max=-1.0,
                        scheduled_lr=0.0,
                        weight_lr_power=weight_lr_power,
                        weight_decay=weight_decay,
                        inner_momentum=inner_momentum,
                        foreach=foreach)
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1, _ = group['betas']
            if train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p to x
                        p.lerp_(end=state['z'].to(p.device), weight=1 - 1 / beta1)
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1, _ = group['betas']
            if not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p to y
                        p.lerp_(end=state['z'].to(p.device), weight=1 - beta1)
                group['train_mode'] = True

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if not self.param_groups[0]['train_mode']:
            raise Exception("Optimizer was not in train mode when step is called. "
                            "Please insert .train() and .eval() calls on the "
                            "optimizer. See documentation for details.")

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps = group['eps']
            beta1, beta2 = group['betas']
            decay = group['weight_decay']
            k = group['k']
            r = group['r']
            warmup_steps = group['warmup_steps']
            weight_lr_power = group['weight_lr_power']
            inner_momentum = group['inner_momentum']

            if k < warmup_steps:
                sched = (k + 1) / warmup_steps
            else:
                sched = 1.0

            bias_correction2 = 1 - beta2 ** (k + 1)
            if inner_momentum != 0:
                bias_correction1 = 1 - inner_momentum ** (k + 1)
            lr = group['lr'] * sched
            group['scheduled_lr'] = lr  # For logging purposes

            lr_max = group['lr_max'] = max(lr, group['lr_max'])

            weight = ((k + 1) ** r) * (lr_max ** weight_lr_power)
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            try:
                ckp1 = weight / weight_sum
            except ZeroDivisionError:
                ckp1 = 0

            active_p = [p for p in group['params'] if p.grad is not None]

            for p in active_p:
                if 'z' not in self.state[p]:
                    self.state[p]['z'] = torch.clone(p, memory_format=torch.preserve_format)
                    self.state[p]['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if inner_momentum != 0:
                        self.state[p]['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

            if group['foreach'] and len(active_p) > 0:
                y, grad, exp_avg_sq, z = zip(*[(p,
                                                p.grad,
                                                self.state[p]['exp_avg_sq'],
                                                self.state[p]['z'])
                                               for p in active_p])

                # Decay the first and second moment running average coefficient
                torch._foreach_mul_(exp_avg_sq, beta2)
                torch._foreach_addcmul_(exp_avg_sq, grad, grad, value=1 - beta2)
                denom = torch._foreach_div(exp_avg_sq, bias_correction2)
                torch._foreach_sqrt_(denom)
                torch._foreach_add_(denom, eps)

                if inner_momentum != 0:
                    exp_avg = tuple(self.state[p]['exp_avg'] for p in active_p)
                    # exp_avg = inner_momentum*exp_avg + (1-inner_momentum)*grad
                    torch._foreach_mul_(exp_avg, inner_momentum)
                    torch._foreach_add_(exp_avg, grad, alpha=1 - inner_momentum)
                    # grad_normalized = (exp_avg / bias_correction1) / denom
                    grad_normalized = torch._foreach_div(exp_avg, bias_correction1)
                    torch._foreach_div_(grad_normalized, denom)
                else:
                    # Normalize grad in-place for memory efficiency
                    torch._foreach_div_(grad, denom)
                    grad_normalized = grad

                # Weight decay calculated at y
                if decay != 0:
                    torch._foreach_add_(grad_normalized, y, alpha=decay)

                # These operations update y in-place,
                # without computing x explicitly.
                torch._foreach_lerp_(y, z, weight=ckp1)
                torch._foreach_add_(y, grad_normalized, alpha=lr * (beta1 * (1 - ckp1) - 1))

                # z step
                torch._foreach_sub_(z, grad_normalized, alpha=lr)
            else:
                for p in active_p:
                    y = p  # Notation to match theory
                    grad = p.grad

                    state = self.state[p]

                    z = state['z']
                    exp_avg_sq = state['exp_avg_sq']

                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)

                    if inner_momentum != 0:
                        exp_avg = state['exp_avg']
                        exp_avg.mul_(inner_momentum).add_(grad, alpha=1 - inner_momentum)
                        grad_normalized = exp_avg.div(bias_correction1).div_(denom)
                    else:
                        # Reuse grad buffer for memory efficiency
                        grad_normalized = grad.div_(denom)

                    # Weight decay calculated at y
                    if decay != 0:
                        grad_normalized.add_(y, alpha=decay)

                    # These operations update y in-place,
                    # without computing x explicitly.
                    y.lerp_(end=z, weight=ckp1)
                    y.add_(grad_normalized, alpha=lr * (beta1 * (1 - ckp1) - 1))

                    # z step
                    z.sub_(grad_normalized, alpha=lr)

            group['k'] = k + 1
        return loss


class RAdamScheduleFree(Optimizer):
    r"""Implements Schedule-Free RAdam, which needs neither a schedule nor a warmup period.

    The update follows :class:`AdamWScheduleFree` with the step size scaled by
    the RAdam rectification term in place of explicit warmup:

    .. math::
       \begin{aligned}
           \rho_\infty &= \frac{2}{1 - \beta_2} - 1, \qquad
               \rho_t = \rho_\infty - \frac{2 t \beta_2^t}{1 - \beta_2^t}, \\
           \gamma_t &= \gamma
               \sqrt{\frac{(\rho_t - 4)(\rho_t - 2)\rho_\infty}
                          {(\rho_\infty - 4)(\rho_\infty - 2)\rho_t}}
               \quad \text{if } \rho_t > 4,
       \end{aligned}

    while for :math:`\rho_t \le 4` the second-moment normalization is skipped
    and the step degenerates to SGD (or, with ``silent_sgd_phase``, only the
    moment estimates are updated).

    Reference: Aaron Defazio, Xingyu Yang, Harsh Mehta, Konstantin Mishchenko,
    Ahmed Khaled, Ashok Cutkosky, "The Road Less Scheduled", NeurIPS 2024.
    https://arxiv.org/abs/2405.15682
    Rectification: Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen,
    Xiaodong Liu, Jianfeng Gao, Jiawei Han, "On the Variance of the Adaptive
    Learning Rate and Beyond", ICLR 2020.
    https://arxiv.org/abs/1908.03265

    Note:
        Call ``optimizer.train()`` before training and ``optimizer.eval()``
        before evaluation or checkpointing, alongside the matching
        ``model.train()`` / ``model.eval()`` calls. Gradients are computed at
        :math:`y_t` while losses should be measured at :math:`x_t`, so the
        parameter buffer must be switched between the two points.
    """

    def __init__(self,
                 params: ParamsT,
                 lr: Union[float, torch.Tensor] = 0.0025,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 r: float = 0.0,
                 weight_lr_power: float = 2.0,
                 foreach: Optional[bool] = hasattr(torch, "_foreach_mul_"),
                 silent_sgd_phase: bool = True
                 ):

        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        r=r,
                        k=0,
                        train_mode=False,
                        weight_sum=0.0,
                        lr_max=-1.0,
                        scheduled_lr=0.0,
                        weight_lr_power=weight_lr_power,
                        weight_decay=weight_decay,
                        foreach=foreach,
                        silent_sgd_phase=silent_sgd_phase)
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            train_mode = group["train_mode"]
            beta1, _ = group["betas"]
            if train_mode:
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        # Set p to x
                        p.lerp_(end=state["z"].to(p.device), weight=1 - 1 / beta1)
                group["train_mode"] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            train_mode = group["train_mode"]
            beta1, _ = group["betas"]
            if not train_mode:
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        # Set p to y
                        p.lerp_(end=state["z"].to(p.device), weight=1 - beta1)
                group["train_mode"] = True

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if not self.param_groups[0]["train_mode"]:
            raise Exception(
                "Optimizer was not in train mode when step is called. "
                "Please insert .train() and .eval() calls on the "
                "optimizer. See documentation for details."
            )

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps = group["eps"]
            beta1, beta2 = group["betas"]
            decay = group["weight_decay"]
            silent_sgd_phase = group["silent_sgd_phase"]
            k = group["k"]  # current steps
            step = k + 1
            r = group['r']
            weight_lr_power = group['weight_lr_power']

            beta2_t = beta2 ** step
            bias_correction2 = 1 - beta2_t

            # maximum length of the approximated SMA
            rho_inf = 2 / (1 - beta2) - 1
            # compute the length of the approximated SMA
            rho_t = rho_inf - 2 * step * beta2_t / bias_correction2
            rect = (
                ((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t)) ** 0.5
                if rho_t > 4.0
                else float(not silent_sgd_phase)
            )

            lr = group["lr"] * rect
            group["scheduled_lr"] = lr  # For logging purposes

            lr_max = group["lr_max"] = max(lr, group["lr_max"])

            weight = (step ** r) * (lr_max ** weight_lr_power)
            weight_sum = group["weight_sum"] = group["weight_sum"] + weight

            try:
                ckp1 = weight / weight_sum
            except ZeroDivisionError:
                ckp1 = 0

            adaptive_y_lr = lr * (beta1 * (1 - ckp1) - 1)
            active_p = [p for p in group["params"] if p.grad is not None]

            for p in active_p:
                if "z" not in self.state[p]:
                    self.state[p]["z"] = torch.clone(p, memory_format=torch.preserve_format)
                    self.state[p]["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

            if group["foreach"] and len(active_p) > 0:
                y, grad, exp_avg_sq, z = zip(
                    *[(p, p.grad, self.state[p]["exp_avg_sq"], self.state[p]["z"]) for p in active_p]
                )

                # Decay the first and second moment running average coefficient
                torch._foreach_mul_(exp_avg_sq, beta2)
                torch._foreach_addcmul_(exp_avg_sq, grad, grad, value=1 - beta2)

                if rho_t > 4.0:
                    # Adam step
                    denom = torch._foreach_div(exp_avg_sq, bias_correction2)
                    torch._foreach_sqrt_(denom)
                    torch._foreach_add_(denom, eps)

                    # Normalize grad in-place for memory efficiency
                    torch._foreach_div_(grad, denom)

                # Weight decay calculated at y
                if decay != 0:
                    torch._foreach_add_(grad, y, alpha=decay)

                # These operations update y in-place,
                # without computing x explicitly.
                torch._foreach_lerp_(y, z, weight=ckp1)
                torch._foreach_add_(y, grad, alpha=adaptive_y_lr)

                # z step
                torch._foreach_sub_(z, grad, alpha=lr)
            else:
                for p in active_p:
                    y = p  # Notation to match theory
                    grad = p.grad

                    state = self.state[p]

                    z = state["z"]
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    if rho_t > 4.0:
                        # Adam step
                        denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)

                        # Reuse grad buffer for memory efficiency
                        grad_normalized = grad.div_(denom)
                    else:
                        # Fall back to SGD (or nothing)
                        grad_normalized = grad

                    # Weight decay calculated at y
                    if decay != 0:
                        grad_normalized.add_(y, alpha=decay)

                    # These operations update y in-place,
                    # without computing x explicitly.
                    y.lerp_(end=z, weight=ckp1)
                    y.add_(grad_normalized, alpha=adaptive_y_lr)

                    # z step
                    z.sub_(grad_normalized, alpha=lr)

            group["k"] = k + 1
        return loss
