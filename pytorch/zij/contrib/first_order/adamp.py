# Adapted from https://github.com/clovaai/AdamP (commit 7f6f038)
# Copyright (c) 2020-present NAVER Corp. Licensed under the MIT license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the AdamP and SGDP optimizers."""

import math

import torch
import torch.nn.functional as F

from ...core.optimizer import Optimizer, ParamsT, required

__all__ = ["AdamP", "SGDP"]


def _channel_view(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1)


def _layer_view(x: torch.Tensor) -> torch.Tensor:
    return x.view(1, -1)


def _cosine_similarity(x, y, eps, view_func):
    x = view_func(x)
    y = view_func(y)
    return F.cosine_similarity(x, y, dim=1, eps=eps).abs_()


def _projection(p, grad, perturb, delta, wd_ratio, eps):
    wd = 1.0
    expand_size = [-1] + [1] * (len(p.shape) - 1)
    for view_func in (_channel_view, _layer_view):
        cosine_sim = _cosine_similarity(grad, p.data, eps, view_func)

        if cosine_sim.max() < delta / math.sqrt(view_func(p.data).size(1)):
            p_n = p.data / view_func(p.data).norm(dim=1).view(expand_size).add_(eps)
            perturb -= p_n * view_func(p_n * perturb).sum(dim=1).view(expand_size)
            wd = wd_ratio
            return perturb, wd

    return perturb, wd


class AdamP(Optimizer):
    r"""Implements AdamP, Adam with a scale-invariant projection step.

    For each layer-weight parameter, the Adam update :math:`p_t` is split into
    its radial and tangential components relative to the weight :math:`\theta`,
    and the radial part is removed whenever the cosine similarity between the
    gradient and the weight is below a threshold (i.e. the weight is treated as
    scale-invariant):

    .. math::
       \begin{aligned}
            m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
            v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
            p_t &= \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \\
            q_t &= \Pi_{\theta_{t-1}}(p_t) \\
            \theta_t &= \theta_{t-1} - \eta \, q_t
       \end{aligned}

    where :math:`\Pi_{\theta}(p) = p - (\hat{\theta} \cdot p)\,\hat{\theta}`
    projects out the component of :math:`p` along the unit weight
    :math:`\hat{\theta}` and ``wd_ratio`` scales the decoupled weight decay on
    the projected parameters.

    Reference: Byeongho Heo, Sanghyuk Chun, Seong Joon Oh, Dongyoon Han,
    Sangdoo Yun, Gyuwan Kim, Youngjung Uh, Jung-Woo Ha, "AdamP: Slowing Down
    the Slowdown for Momentum Optimizers on Scale-invariant Weights", ICLR 2021.
    https://arxiv.org/abs/2006.08217
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        delta: float = 0.1,
        wd_ratio: float = 0.1,
        nesterov: bool = False,
    ) -> None:
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "delta": delta,
            "wd_ratio": wd_ratio,
            "nesterov": nesterov,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                beta1, beta2 = group["betas"]
                nesterov = group["nesterov"]

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                step_size = group["lr"] / bias_correction1

                if nesterov:
                    perturb = (beta1 * exp_avg + (1 - beta1) * grad) / denom
                else:
                    perturb = exp_avg / denom

                wd_ratio = 1.0
                if len(p.shape) > 1:
                    perturb, wd_ratio = _projection(
                        p, grad, perturb, group["delta"], group["wd_ratio"], group["eps"]
                    )

                if group["weight_decay"] > 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"] * wd_ratio)

                p.data.add_(perturb, alpha=-step_size)

        return loss


class SGDP(Optimizer):
    r"""Implements SGDP, SGD with the AdamP scale-invariant projection step.

    The SGD-with-momentum update :math:`p_t` is projected onto the tangent
    space of the weight :math:`\theta` whenever the weight is scale-invariant,
    removing the radial component that drives effective-step-size decay:

    .. math::
       \begin{aligned}
            b_t &= \mu \, b_{t-1} + (1 - \tau) g_t \\
            p_t &= g_t + \mu \, b_t \quad\text{(Nesterov)} \quad\text{or}\quad b_t \\
            q_t &= \Pi_{\theta_{t-1}}(p_t) \\
            \theta_t &= \theta_{t-1} - \eta \, q_t
       \end{aligned}

    where :math:`\mu` is the momentum, :math:`\tau` the dampening, and
    :math:`\Pi_{\theta}(p) = p - (\hat{\theta} \cdot p)\,\hat{\theta}` projects
    out the component of :math:`p` along the unit weight :math:`\hat{\theta}`.

    Reference: Byeongho Heo, Sanghyuk Chun, Seong Joon Oh, Dongyoon Han,
    Sangdoo Yun, Gyuwan Kim, Youngjung Uh, Jung-Woo Ha, "AdamP: Slowing Down
    the Slowdown for Momentum Optimizers on Scale-invariant Weights", ICLR 2021.
    https://arxiv.org/abs/2006.08217
    """

    def __init__(
        self,
        params: ParamsT,
        lr=required,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        eps: float = 1e-8,
        delta: float = 0.1,
        wd_ratio: float = 0.1,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "eps": eps,
            "delta": delta,
            "wd_ratio": wd_ratio,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p.data)

                buf = state["momentum"]
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                if nesterov:
                    d_p = grad + momentum * buf
                else:
                    d_p = buf

                wd_ratio = 1.0
                if len(p.shape) > 1:
                    d_p, wd_ratio = _projection(
                        p, grad, d_p, group["delta"], group["wd_ratio"], group["eps"]
                    )

                if group["weight_decay"] > 0:
                    p.data.mul_(
                        1 - group["lr"] * group["weight_decay"] * wd_ratio / (1 - momentum)
                    )

                p.data.add_(d_p, alpha=-group["lr"])

        return loss
