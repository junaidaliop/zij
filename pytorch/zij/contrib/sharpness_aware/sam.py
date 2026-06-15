# Adapted from https://github.com/davda54/sam (commit 3c3afdb)
# Copyright (c) 2021 David Samuel. Licensed under the MIT license.
# See THIRD_PARTY_NOTICES.md for details.
# mypy: allow-untyped-defs
"""Implementation of the SAM optimizer."""

import torch

from ...core.optimizer import Optimizer

__all__ = ["SAM"]


class SAM(Optimizer):
    r"""Implements SAM, sharpness-aware minimization wrapping a base optimizer.

    .. math::
       \begin{aligned}
       &g_t = \nabla L(\theta_t), \qquad
        \hat{\epsilon}_t = \rho \, \frac{g_t}{\lVert g_t \rVert_2}      \\
       &\theta_{t+1} = \theta_t - \eta \, \nabla L(\theta)
           \big\rvert_{\theta = \theta_t + \hat{\epsilon}_t}
       \end{aligned}

    where :math:`\hat{\epsilon}_t` solves the inner maximization
    :math:`\max_{\lVert \epsilon \rVert_2 \leq \rho} L(\theta_t + \epsilon)`
    to first order, and the gradient at the perturbed point is fed to the
    wrapped base optimizer. With ``adaptive=True`` the perturbation becomes
    the scale-invariant
    :math:`\hat{\epsilon}_t = \rho \, \theta_t^2 g_t / \lVert \theta_t g_t \rVert_2`
    of ASAM (Kwon et al., ICML 2021).

    Reference: Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur,
    "Sharpness-Aware Minimization for Efficiently Improving Generalization",
    ICLR 2021.
    https://arxiv.org/abs/2010.01412

    Note:
        Each step needs two forward-backward passes: either call
        :meth:`first_step`, recompute the loss and gradients, then call
        :meth:`second_step`, or pass :meth:`step` a closure that zeroes
        gradients, computes the loss, and calls ``backward()``.
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, (
            "Sharpness Aware Minimization requires closure, but it was not provided"
        )
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        # put everything on the same device, in case of model parallelism
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
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
