# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa0)
# Copyright (c) Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
"""Implementation of the WSAM optimizer."""

from typing import Callable, Optional

import torch
from torch import nn
from torch.distributed import ReduceOp, all_reduce, is_initialized
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.utils import clip_grad_norm_

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["WSAM"]


def get_global_gradient_norm(
    param_groups: list[dict], device: torch.device
) -> torch.Tensor:
    """Get global gradient norm."""
    norms: list[torch.Tensor] = []
    for group in param_groups or []:
        params: list[torch.Tensor] = group.get("params", []) or []
        adaptive: bool = group.get("adaptive", False)
        for p in params:
            if p.grad is not None:
                norm = ((torch.abs(p) if adaptive else 1.0) * p.grad).norm(p=2).to(
                    device
                )
                norms.append(norm)

    if not norms:
        return torch.tensor(0.0, device=device)

    return torch.norm(torch.stack(norms), p=2)


def disable_running_stats(model: nn.Module) -> None:
    """Disable running stats (momentum) of BatchNorm."""

    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model: nn.Module) -> None:
    """Enable running stats (momentum) of BatchNorm."""

    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


class WSAM(Optimizer):
    r"""Implements WSAM, sharpness-aware minimization with the sharpness weighted as a regularization term.

    .. math::
       \begin{aligned}
            &L^{\mathit{WSAM}}(\theta) = L(\theta) + \frac{\gamma}{1 - \gamma}
                \Bigl( \max_{\lVert \delta \rVert_2 \leq \rho} L(\theta + \delta)
                - L(\theta) \Bigr)                                               \\
            &\delta_t = \rho \, \frac{g_t}{\lVert g_t \rVert_2 + \epsilon}, \qquad
             \tilde{g}_t = \nabla L(\theta_t + \delta_t)                         \\
            &\theta_{t+1} = \theta_t - \eta \, \Bigl( u_t + \frac{\gamma}{1 - \gamma}
                \bigl( \tilde{g}_t - g_t \bigr) \Bigr)
       \end{aligned}

    where :math:`u_t` is the base optimizer update computed from :math:`g_t`.
    With ``decouple=False`` the weighted gradient
    :math:`g_t + \frac{\gamma}{1 - \gamma} (\tilde{g}_t - g_t)` is fed to the
    base optimizer instead.

    Reference: Yun Yue, Jiadi Jiang, Zhiling Ye, Ning Gao, Yongchao Liu, Ke Zhang,
    "Sharpness-Aware Minimization Revisited: Weighted Sharpness as a Regularization Term", KDD 2023.
    https://arxiv.org/abs/2305.15817

    Note:
        WSAM wraps a base optimizer and needs two forward-backward passes per
        step: call :meth:`step` with a closure, or call :meth:`first_step` and
        :meth:`second_step` around the second backward pass. Pass ``model`` so
        BatchNorm running stats are frozen during the second pass.
    """

    def __init__(
        self,
        params: ParamsT,
        base_optimizer: type[Optimizer],
        model: Optional[nn.Module] = None,
        rho: float = 0.05,
        gamma: float = 0.9,
        adaptive: bool = False,
        decouple: bool = True,
        max_norm: Optional[float] = None,
        eps: float = 1e-12,
        **kwargs,
    ):
        if rho < 0.0:
            raise ValueError(f"Invalid rho value: {rho}")
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"Invalid gamma value: {gamma}")

        self.model = model
        self.decouple = decouple
        self.max_norm = max_norm

        alpha: float = gamma / (1.0 - gamma)

        defaults = {
            "rho": rho,
            "alpha": alpha,
            "adaptive": adaptive,
            "sam_eps": eps,
            **kwargs,
        }

        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    def __str__(self) -> str:
        return "WSAM"

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        device = self.param_groups[0]["params"][0].device

        grad_norm = get_global_gradient_norm(self.param_groups, device)

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + group["sam_eps"])

            for p in group["params"]:
                if p.grad is None:
                    continue

                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)

                p.add_(e_w)

                self.state[p]["e_w"] = e_w

                if is_initialized():
                    all_reduce(p.grad, op=ReduceOp.AVG)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                self.state[p]["grad"] = p.grad.clone()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if is_initialized():
                    all_reduce(p.grad, ReduceOp.AVG)

                p.add_(self.state[p]["e_w"], alpha=-1.0)

        if self.max_norm is not None:
            parameters = (
                self.model.parameters()
                if self.model is not None
                else [p for group in self.param_groups for p in group["params"]]
            )
            clip_grad_norm_(parameters, self.max_norm)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if not self.decouple:
                    p.grad.mul_(group["alpha"]).add_(
                        self.state[p]["grad"], alpha=1.0 - group["alpha"]
                    )
                else:
                    self.state[p]["sharpness"] = p.grad.clone() - self.state[p]["grad"]
                    p.grad.mul_(0.0).add_(self.state[p]["grad"], alpha=1.0)

        self.base_optimizer.step()

        if self.decouple:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    p.add_(
                        self.state[p]["sharpness"],
                        alpha=-group["lr"] * group["alpha"],
                    )

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if closure is None:
            raise ValueError(
                "WSAM requires a closure to reevaluate the model after the perturbation"
            )

        closure = torch.enable_grad()(closure)

        if self.model is not None:
            enable_running_stats(self.model)
        loss = closure()

        self.first_step(zero_grad=True)

        if self.model is not None:
            disable_running_stats(self.model)
        closure()

        self.second_step()

        return loss

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
