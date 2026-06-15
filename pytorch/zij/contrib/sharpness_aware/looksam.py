# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa0)
# Copyright (c) kozistr (Hyeongchan Kim). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
"""Implementation of the LookSAM optimizer."""

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["LookSAM"]


def _centralize_gradient(grad: Tensor) -> None:
    if grad.dim() > 1:
        grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))


def _global_gradient_norm(
    param_groups: list[dict[str, Any]], device: torch.device
) -> Tensor:
    norms: list[Tensor] = []
    for group in param_groups:
        for p in group["params"]:
            if p.grad is not None:
                norms.append(
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(device)
                )

    if not norms:
        return torch.tensor(0.0, device=device)

    return torch.norm(torch.stack(norms), p=2)


class LookSAM(Optimizer):
    r"""Implements LookSAM, a Sharpness-Aware Minimization variant that takes the ascent step only once every :math:`k` steps.

    .. math::
       \begin{aligned}
       &\text{if } t \bmod k = 0: \\
       &\quad \epsilon_t = \rho \, \frac{g_t}{\lVert g_t \rVert_2}, \qquad
          g_s = \nabla_{\theta} L(\theta_t + \epsilon_t) \\
       &\quad g_v = g_s - \lVert g_s \rVert_2
          \frac{g_t \cdot g_s}{\lVert g_t \rVert_2 \lVert g_s \rVert_2}
          \frac{g_t}{\lVert g_t \rVert_2} \\
       &\text{otherwise:} \\
       &\quad g_s = g_t + \alpha
          \frac{\lVert g_t \rVert_2}{\lVert g_v \rVert_2} g_v \\
       &\theta_{t+1} = \theta_t - \eta \, g_s
       \end{aligned}

    where :math:`g_t` is the minibatch gradient at :math:`\theta_t` and the
    descent update with :math:`g_s` is delegated to the wrapped base optimizer.

    Note:
        Gradients must be computed before calling :meth:`step`, which takes a
        closure that re-evaluates the loss and calls ``backward()`` at the
        perturbed point; the closure runs only on refresh steps
        (``get_step() % k == 0``). Alternatively, call :meth:`first_step` and
        :meth:`second_step` explicitly, running the second forward-backward
        pass only on refresh steps. Following the upstream implementation,
        the :math:`g_v` decomposition and the reuse-step scaling are applied
        per parameter tensor rather than over the global parameter vector.
        The refresh schedule reads a ``step`` counter from the base
        optimizer, so the base should be an optimizer that tracks per-step
        state such as Adam or AdamW; with a stateless base every step
        refreshes, which is correct but saves no computation.

    Reference: Yong Liu, Siqi Mai, Xiangning Chen, Cho-Jui Hsieh, Yang You,
    "Towards Efficient and Scalable Sharpness-Aware Minimization", CVPR 2022.
    https://arxiv.org/abs/2203.02714
    """

    def __init__(
        self,
        params: ParamsT,
        base_optimizer: type[Optimizer],
        rho: float = 0.1,
        k: int = 10,
        alpha: float = 0.7,
        adaptive: bool = False,
        use_gc: bool = False,
        perturb_eps: float = 1e-12,
        **kwargs,
    ) -> None:
        if not 0.0 <= rho:
            raise ValueError(f"Invalid rho value: {rho}")
        if not 0 < k:
            raise ValueError(f"Invalid k value: {k}")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if not 0.0 <= perturb_eps:
            raise ValueError(f"Invalid perturb_eps value: {perturb_eps}")

        self.k = k
        self.alpha = alpha
        self.use_gc = use_gc
        self.perturb_eps = perturb_eps

        defaults: dict[str, Any] = {"rho": rho, "adaptive": adaptive}
        defaults.update(kwargs)

        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    def get_step(self):
        if "step" in self.param_groups[0]:
            return self.param_groups[0]["step"]
        if self.base_optimizer.state:
            return next(iter(self.base_optimizer.state.values()))["step"]
        return 0

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        if self.get_step() % self.k != 0:
            return

        device = self.param_groups[0]["params"][0].device

        grad_norm = _global_gradient_norm(self.param_groups, device).add_(
            self.perturb_eps
        )

        for i, group in enumerate(self.param_groups):
            scale = group["rho"] / grad_norm

            for j, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                grad = p.grad
                if self.use_gc:
                    _centralize_gradient(grad)

                self.state[p]["old_p"] = p.clone()
                self.state[f"old_grad_p_{i}{j}"]["old_grad_p"] = grad.clone()

                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * grad
                    * scale.to(p)
                )

                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        step = self.get_step()

        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                grad = p.grad
                grad_norm = grad.norm(p=2)

                if step % self.k == 0:
                    old_grad_p = self.state[f"old_grad_p_{i}{j}"]["old_grad_p"]

                    g_grad_norm = old_grad_p / old_grad_p.norm(p=2)
                    g_s_grad_norm = grad / grad_norm

                    self.state[f"gv_{i}{j}"]["gv"] = torch.sub(
                        grad,
                        grad_norm * torch.sum(g_grad_norm * g_s_grad_norm) * g_grad_norm,
                    )
                else:
                    gv = self.state[f"gv_{i}{j}"]["gv"]
                    grad.add_(grad_norm / (gv.norm(p=2) + 1e-8) * gv, alpha=self.alpha)

                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> None:
        if closure is None:
            raise ValueError(
                "LookSAM requires a closure that re-evaluates the loss and "
                "calls backward()"
            )

        if self.get_step() % self.k == 0:
            self.first_step(zero_grad=True)
            with torch.enable_grad():
                closure()

        self.second_step()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
