# Adapted from https://github.com/Liuhong99/Sophia (commit a7e1572)
# Copyright (c) 2023 Hong Liu. Licensed under the MIT license.
# See THIRD_PARTY_NOTICES.md for details.
"""Implementation of the Sophia optimizer."""

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["SophiaG"]


class SophiaG(Optimizer):
    r"""Implements Sophia (Gauss-Newton-Bartlett variant), a second-order
    clipped stochastic optimizer.

    Sophia preconditions the gradient with a moving average of a light-weight
    diagonal Hessian estimate and clips the result element-wise, which bounds
    the worst-case update along any coordinate. With first moment :math:`m_t`,
    diagonal Hessian estimate :math:`h_t`, learning rate :math:`\eta`, decay
    rates :math:`\beta_1`, :math:`\beta_2`, and pre-conditioner coefficient
    :math:`\rho` (the paper's :math:`\gamma`), with the per-coordinate clip
    applied to magnitude 1:

    .. math::
       \begin{aligned}
       m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, g_t \\
       h_t &= \beta_2 h_{t-k} + (1 - \beta_2)\, \hat{h}_t \\
       \theta_t &= (1 - \eta\lambda)\, \theta_{t-1} - \eta\,
           \mathrm{clip}\!\left(
               \frac{m_t}{\rho\, B\, h_t + \epsilon},\, 1
           \right)
       \end{aligned}

    where :math:`\lambda` is the decoupled ``weight_decay`` and :math:`B` is the
    ``bs`` (batch size) passed to :meth:`step`. The Hessian estimate
    :math:`h_t` is refreshed every :math:`k` steps by :meth:`update_hessian`.
    The Gauss-Newton-Bartlett estimator forms :math:`\hat{h}_t` from the
    per-coordinate squared gradient of a loss evaluated on labels sampled from
    the model's own predictive distribution; the batch-size factor :math:`B` is
    applied here in the denominator rather than folded into
    :math:`\hat{h}_t`, following the official implementation. The clip operates
    per coordinate, so the effective step never exceeds :math:`\eta` in
    magnitude.

    Note: Sophia requires a periodic Hessian refresh. Call
    :meth:`update_hessian` every ``k`` steps after a backward pass on a sampled
    loss (a closure), then call :meth:`step`. The ``bs`` argument to
    :meth:`step` is the batch size used to scale the estimator. Until the first
    :meth:`update_hessian` call the estimate is zero and every update saturates
    the clip, reducing the step to :math:`-\eta\,\mathrm{sign}(m_t)`.

    Reference: Hong Liu, Zhiyuan Li, David Hall, Percy Liang, Tengyu Ma,
    "Sophia: A Scalable Stochastic Second-order Optimizer for Language Model
    Pre-training", ICLR 2024.
    https://arxiv.org/abs/2305.14342
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-4,
        betas=(0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 1e-1,
        *,
        maximize: bool = False,
        capturable: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if rho < 0.0:
            raise ValueError(f"Invalid rho parameter: {rho}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            rho=rho,
            weight_decay=weight_decay,
            maximize=maximize,
            capturable=capturable,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("capturable", False)
        state_values = list(self.state.values())
        step_is_tensor = len(state_values) != 0 and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    def _init_state(self, group, p):
        state = self.state[p]
        if len(state) == 0:
            state["step"] = (
                torch.zeros((1,), dtype=torch.float, device=p.device)
                if group["capturable"]
                else torch.tensor(0.0)
            )
            state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["hessian"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        if "hessian" not in state:
            state["hessian"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        return state

    @torch.no_grad()
    def update_hessian(self):
        """Refresh the diagonal Hessian estimate from the current gradients.

        Call after a backward pass on a loss sampled for the Gauss-Newton-
        Bartlett estimator, periodically (every ``k`` steps).
        """
        for group in self.param_groups:
            _, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self._init_state(group, p)
                state["hessian"].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

    @torch.no_grad()
    def step(self, closure=None, bs=5120):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            rho = group["rho"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            maximize = group["maximize"]
            capturable = group["capturable"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("SophiaG does not support sparse gradients")

                state = self._init_state(group, p)
                exp_avg = state["exp_avg"]
                hess = state["hessian"]
                step_t = state["step"]

                grad = p.grad if not maximize else -p.grad
                param = p

                if capturable:
                    batch_size = (
                        torch.ones((1,), dtype=torch.float, device=p.device) * bs
                    )
                else:
                    batch_size = bs

                if torch.is_complex(param):
                    grad = torch.view_as_real(grad)
                    exp_avg = torch.view_as_real(exp_avg)
                    hess = torch.view_as_real(hess)
                    param = torch.view_as_real(param)

                step_t += 1

                param.mul_(1 - lr * weight_decay)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                ratio = (exp_avg.abs() / (rho * batch_size * hess + 1e-15)).clamp(
                    None, 1
                )
                param.addcmul_(exp_avg.sign(), ratio, value=-lr)

        return loss
