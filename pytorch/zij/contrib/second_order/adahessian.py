# Adapted from https://github.com/amirgholami/adahessian (commit 85ebc00)
# Copyright (c) 2020 Amir Gholaminejad, Zhewei Yao. Licensed under the MIT license.
# See THIRD_PARTY_NOTICES.md for details.
"""Implementation of the AdaHessian optimizer."""

import math

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["Adahessian", "AdaHessian"]


class Adahessian(Optimizer):
    r"""Implements AdaHessian, an adaptive second-order optimizer.

    AdaHessian replaces the squared-gradient denominator of Adam with a running
    average of the squared diagonal of the Hessian, estimated with a
    Hutchinson matrix-free probe. For each step a Rademacher vector :math:`z`
    (entries :math:`\pm 1`) is drawn and the Hessian-vector product
    :math:`H_t z` is formed by differentiating :math:`g_t^\top z`. The
    per-element magnitude :math:`|H_t z|` is then block-averaged to reduce its
    variance, giving the block-averaged diagonal estimate :math:`D_t^{(s)}`.
    With first moment :math:`m_t`, second moment :math:`v_t` over
    :math:`D_t^{(s)}`, learning rate :math:`\eta`, decay rates :math:`\beta_1`,
    :math:`\beta_2`, and Hessian power :math:`k`:

    .. math::
       \begin{aligned}
       D_t^{(s)} &= \frac{1}{b} \sum_{\text{block}} |H_t z|,
           \qquad z_i \in \{-1, +1\} \\
       m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, g_t \\
       v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, \bigl(D_t^{(s)}\bigr)^2 \\
       \hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \qquad
           \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
       \theta_t &= \theta_{t-1} - \eta \left(
           \frac{\hat{m}_t}{\hat{v}_t^{\,k/2} + \epsilon}
           + \lambda\, \theta_{t-1} \right)
       \end{aligned}

    where :math:`\lambda` is the ``weight_decay`` and :math:`b` is the number of
    elements in each structured block. The per-element magnitude
    :math:`|H_t z|` is averaged (not the signed product) over each block of
    size :math:`b`: a 2D Conv kernel is averaged over its spatial extent,
    matching the block-diagonal averaging of the paper. Setting :math:`k = 1`
    recovers the standard Hessian power; :math:`k = 0.5` is a milder
    preconditioner.

    Note: AdaHessian needs the Hessian-vector product, so the gradients passed
    to :meth:`step` must carry an autograd graph. Call ``loss.backward(
    create_graph=True)`` before :meth:`step` (or pass a closure that does so).
    Without ``create_graph=True`` the gradients have no ``grad_fn`` and
    :meth:`step` raises. Sparse gradients are not supported.

    Reference: Zhewei Yao, Amir Gholami, Sheng Shen, Mustafa Mustafa,
    Kurt Keutzer, Michael W. Mahoney, "ADAHESSIAN: An Adaptive Second Order
    Optimizer for Machine Learning", AAAI 2021.
    https://arxiv.org/abs/2006.00719
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 0.15,
        betas=(0.9, 0.999),
        eps: float = 1e-4,
        weight_decay: float = 0.0,
        hessian_power: float = 1.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError(f"Invalid Hessian power value: {hessian_power}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            hessian_power=hessian_power,
        )
        super().__init__(params, defaults)

    def get_trace(self, params, grads):
        """Estimate the diagonal of the Hessian via a Hutchinson probe.

        Forms the Hessian-vector product with a Rademacher vector and returns
        the per-parameter diagonal estimate, spatially averaged within blocks.
        """
        for i, grad in enumerate(grads):
            if grad.grad_fn is None:
                raise RuntimeError(
                    f"Gradient tensor {i} does not have grad_fn. When calling "
                    "loss.backward(), make sure the option create_graph is set "
                    "to True."
                )

        v = [2 * torch.randint_like(p, high=2) - 1 for p in params]

        hvs = torch.autograd.grad(
            grads,
            params,
            grad_outputs=v,
            only_inputs=True,
            retain_graph=True,
        )

        hutchinson_trace = []
        for hv in hvs:
            param_size = hv.size()
            if len(param_size) <= 2:
                # 0/1/2D tensor: |hv * v| = |hv| since v is +/-1, block size 1.
                tmp_output = hv.abs()
            elif len(param_size) == 4:
                # Conv kernel: average over the spatial dimensions.
                tmp_output = torch.mean(hv.abs(), dim=[2, 3], keepdim=True)
            else:
                raise RuntimeError(
                    "AdaHessian does not support parameters of shape "
                    f"{tuple(param_size)}; supported ranks are 0, 1, 2, 4."
                )
            hutchinson_trace.append(tmp_output)

        return hutchinson_trace

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss. The backward pass it triggers must use
                ``create_graph=True``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        params = []
        groups = []
        grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("AdaHessian does not support sparse gradients")
                params.append(p)
                groups.append(group)
                grads.append(p.grad)

        hut_traces = self.get_trace(params, grads)

        for p, group, grad, hut_trace in zip(params, groups, grads, hut_traces):
            state = self.state[p]

            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_hessian_diag_sq"] = torch.zeros_like(p)

            exp_avg = state["exp_avg"]
            exp_hessian_diag_sq = state["exp_hessian_diag_sq"]
            beta1, beta2 = group["betas"]

            state["step"] += 1

            exp_avg.mul_(beta1).add_(grad.detach(), alpha=1 - beta1)
            exp_hessian_diag_sq.mul_(beta2).addcmul_(
                hut_trace, hut_trace, value=1 - beta2
            )

            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]

            k = group["hessian_power"]
            denom = (
                exp_hessian_diag_sq.sqrt() ** k / math.sqrt(bias_correction2) ** k
            ).add_(group["eps"])

            with torch.no_grad():
                p.add_(
                    exp_avg / bias_correction1 / denom + group["weight_decay"] * p,
                    alpha=-group["lr"],
                )

        return loss


AdaHessian = Adahessian
