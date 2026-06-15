# Adapted from https://github.com/Luolc/AdaBound (commit 2e928c3)
# Copyright (c) 2019 Liangchen Luo. Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the AdaBound and AdaBoundW optimizers."""

import math

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["AdaBound", "AdaBoundW"]


class AdaBound(Optimizer):
    r"""Implements AdaBound, Adam with a dynamic bound on the learning rate.

    .. math::
       \begin{aligned}
            m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t                            \\
            v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2                          \\
            \hat{\eta}_t &= \mathrm{Clip}\!\left(
                \frac{\eta \sqrt{1 - \beta_2^t}}{(1 - \beta_1^t)
                    (\sqrt{v_t} + \epsilon)},
                \eta_l(t), \eta_u(t)\right)                                       \\
            \eta_l(t) &= \eta^* \left(1 - \frac{1}{\gamma t + 1}\right),
                \quad \eta_u(t) = \eta^* \left(1 + \frac{1}{\gamma t}\right)      \\
            \theta_t &= \theta_{t-1} - \hat{\eta}_t \odot m_t
       \end{aligned}

    where :math:`\eta^*` is the final (SGD) learning rate. The lower and upper
    bounds converge to :math:`\eta^*` as :math:`t \to \infty`, so AdaBound
    transitions smoothly from Adam to SGD.

    Reference: Liangchen Luo, Yuanhao Xiong, Yan Liu, Xu Sun, "Adaptive Gradient
    Methods with Dynamic Bound of Learning Rate", ICLR 2019.
    https://arxiv.org/abs/1902.09843
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        final_lr: float = 0.1,
        gamma: float = 1e-3,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsbound: bool = False,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= final_lr:
            raise ValueError(f"Invalid final learning rate: {final_lr}")
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"Invalid gamma parameter: {gamma}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "final_lr": final_lr,
            "gamma": gamma,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsbound": amsbound,
        }
        super().__init__(params, defaults)

        self.base_lrs = [group["lr"] for group in self.param_groups]

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsbound", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "AdaBound does not support sparse gradients"
                    )
                amsbound = group["amsbound"]

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if amsbound:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsbound:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = (
                    group["lr"] * math.sqrt(bias_correction2) / bias_correction1
                )

                # lr_scheduler cannot affect final_lr; rescale by base_lr so any
                # learning-rate decay applied to group["lr"] carries through.
                final_lr = group["final_lr"] * group["lr"] / base_lr
                lower_bound = final_lr * (1 - 1 / (group["gamma"] * state["step"] + 1))
                upper_bound = final_lr * (1 + 1 / (group["gamma"] * state["step"]))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)

                p.add_(-step_size)

        return loss


class AdaBoundW(Optimizer):
    r"""Implements AdaBound with decoupled weight decay (AdamW-style).

    The adaptive update matches :class:`AdaBound`; weight decay is applied
    directly to the parameters rather than added to the gradient.

    .. math::
       \begin{aligned}
            m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t                            \\
            v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2                          \\
            \hat{\eta}_t &= \mathrm{Clip}\!\left(
                \frac{\eta \sqrt{1 - \beta_2^t}}{(1 - \beta_1^t)
                    (\sqrt{v_t} + \epsilon)},
                \eta_l(t), \eta_u(t)\right)                                       \\
            \theta_t &= \theta_{t-1} - \hat{\eta}_t \odot m_t - \lambda \theta_{t-1}
       \end{aligned}

    where :math:`\lambda` is the weight decay and :math:`\eta_l(t)`,
    :math:`\eta_u(t)` are the AdaBound bounds.

    Reference: Liangchen Luo, Yuanhao Xiong, Yan Liu, Xu Sun, "Adaptive Gradient
    Methods with Dynamic Bound of Learning Rate", ICLR 2019.
    https://arxiv.org/abs/1902.09843
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        final_lr: float = 0.1,
        gamma: float = 1e-3,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsbound: bool = False,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= final_lr:
            raise ValueError(f"Invalid final learning rate: {final_lr}")
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"Invalid gamma parameter: {gamma}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "final_lr": final_lr,
            "gamma": gamma,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsbound": amsbound,
        }
        super().__init__(params, defaults)

        self.base_lrs = [group["lr"] for group in self.param_groups]

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsbound", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "AdaBoundW does not support sparse gradients"
                    )
                amsbound = group["amsbound"]

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if amsbound:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsbound:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = (
                    group["lr"] * math.sqrt(bias_correction2) / bias_correction1
                )

                # lr_scheduler cannot affect final_lr; rescale by base_lr so any
                # learning-rate decay applied to group["lr"] carries through.
                final_lr = group["final_lr"] * group["lr"] / base_lr
                lower_bound = final_lr * (1 - 1 / (group["gamma"] * state["step"] + 1))
                upper_bound = final_lr * (1 + 1 / (group["gamma"] * state["step"]))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)

                if group["weight_decay"] != 0:
                    decayed_weights = torch.mul(p, group["weight_decay"])
                    p.add_(-step_size)
                    p.sub_(decayed_weights)
                else:
                    p.add_(-step_size)

        return loss
