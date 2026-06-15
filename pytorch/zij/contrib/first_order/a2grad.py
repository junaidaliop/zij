# Adapted from https://github.com/jettify/pytorch-optimizer (commit 19c3e41)
# Copyright (c) 2020 Nikolay Novik. Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
"""Implementation of the A2Grad optimizers (Uni, Inc, Exp variants)."""

import copy
import math
from typing import Optional

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["A2GradUni", "A2GradInc", "A2GradExp"]


class A2GradUni(Optimizer):
    r"""Implements A2Grad (uniform variant), adaptive accelerated SGD.

    Accelerated stochastic gradient descent with a diagonal adaptive term.
    Three sequences are coupled per step: a gradient-evaluation point
    :math:`x_k`, an averaged point, and the iterate :math:`\theta_k`. The
    step coefficient mixes the Lipschitz term :math:`\gamma_k` with an
    adaptive accumulation :math:`h_k` of the gradient deviation from its
    running average:

    .. math::
       \begin{aligned}
       \gamma_k &= \frac{2 L}{k + 1} \\
       \bar{g}_k &= \frac{1}{k + 1} \sum_{i=0}^{k} g_i \\
       \delta_k &= g_k - \bar{g}_k \\
       v_k &= v_{k-1} + \lVert \delta_k \rVert^2,
            \qquad h_k = \sqrt{v_k} \\
       \alpha_k &= \frac{2}{k + 3},
            \qquad c_k = \frac{1}{\gamma_k + \beta h_k} \\
       x_{k+1} &= x_k - c_k\, g_k \\
       \theta_{k+1} &= (1 - \alpha_k)\,\theta_k + \alpha_k\, x_{k+1}
                       - (1 - \alpha_k)\,\alpha_{k-1}\, c_k\, g_k
       \end{aligned}

    Reference: Qi Deng, Yi Cheng, Guanghui Lan, "Optimal Adaptive and
    Accelerated Stochastic Gradient Descent", arXiv 2018.
    https://arxiv.org/abs/1810.00553
    """

    def __init__(
        self,
        params: ParamsT,
        lr: Optional[float] = None,
        beta: float = 10.0,
        lips: float = 10.0,
    ) -> None:
        if beta < 0.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if lips < 0.0:
            raise ValueError(f"Invalid lips value: {lips}")

        # lr is unused by this optimizer; it is kept so learning-rate
        # schedulers do not fail when wrapping the optimizer.
        defaults = {"beta": beta, "lips": lips, "lr": lr}
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
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["alpha_k"] = 1
                    state["v_k"] = 0
                    state["avg_grad"] = copy.deepcopy(grad)
                    state["x_k"] = copy.deepcopy(p)

                gamma_k = 2 * group["lips"] / (state["step"] + 1)

                avg_grad = state["avg_grad"]
                avg_grad.mul_(state["step"])
                avg_grad.add_(grad)
                avg_grad.div_(state["step"] + 1)

                delta_k = torch.add(grad, avg_grad, alpha=-1)

                state["v_k"] += torch.sum(delta_k * delta_k).item()

                h_k = math.sqrt(state["v_k"])
                alpha_k_1 = 2 / (state["step"] + 3)
                coef = 1 / (gamma_k + group["beta"] * h_k)
                x_k_1 = state["x_k"]
                x_k_1.add_(grad, alpha=-coef)

                p.mul_(1 - alpha_k_1)
                p.add_(x_k_1, alpha=alpha_k_1)
                p.add_(grad, alpha=-(1 - alpha_k_1) * state["alpha_k"] * coef)

                state["alpha_k"] = alpha_k_1
                state["step"] += 1

        return loss


class A2GradInc(Optimizer):
    r"""Implements A2Grad (incremental variant), adaptive accelerated SGD.

    Same accelerated coupling as :class:`A2GradUni`, but the adaptive
    accumulator is discounted incrementally before each gradient deviation
    is added:

    .. math::
       v_k = \left(\frac{k}{k + 1}\right)^2 v_{k-1}
             + \lVert \delta_k \rVert^2

    Reference: Qi Deng, Yi Cheng, Guanghui Lan, "Optimal Adaptive and
    Accelerated Stochastic Gradient Descent", arXiv 2018.
    https://arxiv.org/abs/1810.00553
    """

    def __init__(
        self,
        params: ParamsT,
        lr: Optional[float] = None,
        beta: float = 10.0,
        lips: float = 10.0,
    ) -> None:
        if beta < 0.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if lips < 0.0:
            raise ValueError(f"Invalid lips value: {lips}")

        # lr is unused by this optimizer; it is kept so learning-rate
        # schedulers do not fail when wrapping the optimizer.
        defaults = {"beta": beta, "lips": lips, "lr": lr}
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
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["alpha_k"] = 1
                    state["v_k"] = 0
                    state["avg_grad"] = copy.deepcopy(grad)
                    state["x_k"] = copy.deepcopy(p)

                gamma_k = 2 * group["lips"] / (state["step"] + 1)

                avg_grad = state["avg_grad"]
                avg_grad.mul_(state["step"])
                avg_grad.add_(grad)
                avg_grad.div_(state["step"] + 1)

                delta_k = torch.add(grad, avg_grad, alpha=-1)

                state["v_k"] *= (state["step"] / (state["step"] + 1)) ** 2
                state["v_k"] += torch.sum(delta_k * delta_k).item()

                h_k = math.sqrt(state["v_k"])
                alpha_k_1 = 2 / (state["step"] + 3)
                coef = 1 / (gamma_k + group["beta"] * h_k)
                x_k_1 = state["x_k"]
                x_k_1.add_(grad, alpha=-coef)

                p.mul_(1 - alpha_k_1)
                p.add_(x_k_1, alpha=alpha_k_1)
                p.add_(grad, alpha=-(1 - alpha_k_1) * state["alpha_k"] * coef)

                state["alpha_k"] = alpha_k_1
                state["step"] += 1

        return loss


class A2GradExp(Optimizer):
    r"""Implements A2Grad (exponential variant), adaptive accelerated SGD.

    Same accelerated coupling as :class:`A2GradUni`, but the adaptive
    accumulator is an exponential moving average of the gradient deviation,
    kept monotone and scaled by the step count:

    .. math::
       \begin{aligned}
       \tilde{v}_k &= \rho\, \tilde{v}_{k-1}
                      + (1 - \rho)\, \lVert \delta_k \rVert^2 \\
       v_k &= \max(v_{k-1}, \tilde{v}_k),
            \qquad h_k = \sqrt{(k + 1)\, v_k}
       \end{aligned}

    Reference: Qi Deng, Yi Cheng, Guanghui Lan, "Optimal Adaptive and
    Accelerated Stochastic Gradient Descent", arXiv 2018.
    https://arxiv.org/abs/1810.00553
    """

    def __init__(
        self,
        params: ParamsT,
        lr: Optional[float] = None,
        beta: float = 10.0,
        lips: float = 10.0,
        rho: float = 0.5,
    ) -> None:
        if beta < 0.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if lips < 0.0:
            raise ValueError(f"Invalid lips value: {lips}")
        if rho < 0.0 or rho > 1.0:
            raise ValueError(f"Invalid rho value: {rho}")

        # lr is unused by this optimizer; it is kept so learning-rate
        # schedulers do not fail when wrapping the optimizer.
        defaults = {"beta": beta, "lips": lips, "rho": rho, "lr": lr}
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
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["alpha_k"] = 1
                    state["v_k"] = 0
                    state["avg_grad"] = copy.deepcopy(grad)
                    state["x_k"] = copy.deepcopy(p)

                gamma_k = 2 * group["lips"] / (state["step"] + 1)

                avg_grad = state["avg_grad"]
                avg_grad.mul_(state["step"])
                avg_grad.add_(grad)
                avg_grad.div_(state["step"] + 1)

                delta_k = torch.add(grad, avg_grad, alpha=-1)

                if state["step"] == 0:
                    state["v_kk"] = torch.sum(delta_k * delta_k).item()
                else:
                    state["v_kk"] *= group["rho"]
                    state["v_kk"] += (1 - group["rho"]) * torch.sum(delta_k * delta_k).item()
                state["v_k"] = max([state["v_kk"], state["v_k"]])

                h_k = math.sqrt((state["step"] + 1) * state["v_k"])

                alpha_k_1 = 2 / (state["step"] + 3)

                coef = -1 / (gamma_k + group["beta"] * h_k)
                x_k_1 = state["x_k"]
                x_k_1.add_(grad, alpha=coef)

                p.mul_(1 - alpha_k_1)
                p.add_(x_k_1, alpha=alpha_k_1)
                p.add_(grad, alpha=(1 - alpha_k_1) * state["alpha_k"] * coef)

                state["alpha_k"] = alpha_k_1
                state["step"] += 1

        return loss
