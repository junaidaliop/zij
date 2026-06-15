# Adapted from https://github.com/yangluo7/CAME (commit e77c5c0)
# Copyright (c) 2023 Yang Luo. Licensed under the MIT license.
# See THIRD_PARTY_NOTICES.md for details.
"""Implementation of the CAME optimizer."""

import torch

from ...core.optimizer import Optimizer

__all__ = ["CAME"]


class CAME(Optimizer):
    r"""Implements CAME, a confidence-guided variant of Adafactor-style factored optimization.

    .. math::
       \begin{aligned}
       r_t &= \beta_2 r_{t-1} + (1 - \beta_2)\,
              \bigl(g_t^2 + \epsilon_1 1_n 1_m^\top\bigr) 1_m \\
       c_t &= \beta_2 c_{t-1} + (1 - \beta_2)\,
              1_n^\top \bigl(g_t^2 + \epsilon_1 1_n 1_m^\top\bigr) \\
       v_t &= r_t c_t / (1_n^\top r_t) \\
       u_t &= g_t / \sqrt{v_t} \\
       \hat{u}_t &= u_t / \max\bigl(1, \mathrm{RMS}(u_t) / d\bigr) \\
       m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, \hat{u}_t \\
       U_t &= (\hat{u}_t - m_t)^2 \\
       R_t &= \beta_3 R_{t-1} + (1 - \beta_3)\,
              \bigl(U_t + \epsilon_2 1_n 1_m^\top\bigr) 1_m \\
       C_t &= \beta_3 C_{t-1} + (1 - \beta_3)\,
              1_n^\top \bigl(U_t + \epsilon_2 1_n 1_m^\top\bigr) \\
       S_t &= R_t C_t / (1_n^\top R_t) \\
       \theta_t &= \theta_{t-1} - \frac{\eta}{\sqrt{S_t}}\, m_t
       \end{aligned}

    where :math:`d` is the clipping threshold, :math:`\epsilon_1` and
    :math:`\epsilon_2` are the regularization constants given by ``eps``, and
    :math:`(\beta_1, \beta_2, \beta_3)` are the decay rates of the update,
    square-gradient, and instability moving averages. Parameters with fewer
    than two dimensions are not factored and skip the confidence-guided
    correction.

    Reference: Yang Luo, Xiaozhe Ren, Zangwei Zheng, Zhuo Jiang, Xin Jiang,
    Yang You, "CAME: Confidence-guided Adaptive Memory Efficient Optimization",
    ACL 2023.
    https://arxiv.org/abs/2307.02047
    """

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-16),
        clip_threshold=1.0,
        betas=(0.9, 0.999, 0.9999),
        weight_decay=0.0,
    ):
        if lr is None or lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not all(0.0 <= beta <= 1.0 for beta in betas):
            raise ValueError(f"Invalid betas: {betas}")

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            betas=betas,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

    def _get_options(self, param_shape):
        factored = len(param_shape) >= 2
        return factored

    def _rms(self, tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (
            (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
            .rsqrt_()
            .unsqueeze(-1)
        )
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("CAME does not support sparse gradients.")

                state = self.state[p]
                grad_shape = grad.shape

                factored = self._get_options(grad_shape)
                # State Initialization
                if len(state) == 0:
                    state["step"] = 0

                    state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).type_as(grad)
                        state["exp_avg_sq_col"] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:]
                        ).type_as(grad)

                        state["exp_avg_res_row"] = torch.zeros(grad_shape[:-1]).type_as(grad)
                        state["exp_avg_res_col"] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:]
                        ).type_as(grad)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["RMS"] = 0

                state["step"] += 1
                state["RMS"] = self._rms(p.data)

                update = (grad**2) + group["eps"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(group["betas"][1]).add_(
                        update.mean(dim=-1), alpha=1.0 - group["betas"][1]
                    )
                    exp_avg_sq_col.mul_(group["betas"][1]).add_(
                        update.mean(dim=-2), alpha=1.0 - group["betas"][1]
                    )

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(group["betas"][1]).add_(update, alpha=1.0 - group["betas"][1])
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_(
                    (self._rms(update) / group["clip_threshold"]).clamp_(min=1.0)
                )

                exp_avg = state["exp_avg"]
                exp_avg.mul_(group["betas"][0]).add_(update, alpha=1 - group["betas"][0])

                # Confidence-guided strategy
                # Calculation of instability
                res = (update - exp_avg)**2 + group["eps"][1]

                if factored:
                    exp_avg_res_row = state["exp_avg_res_row"]
                    exp_avg_res_col = state["exp_avg_res_col"]

                    exp_avg_res_row.mul_(group["betas"][2]).add_(
                        res.mean(dim=-1), alpha=1.0 - group["betas"][2]
                    )
                    exp_avg_res_col.mul_(group["betas"][2]).add_(
                        res.mean(dim=-2), alpha=1.0 - group["betas"][2]
                    )

                    # Approximation of exponential moving average of instability
                    res_approx = self._approx_sq_grad(exp_avg_res_row, exp_avg_res_col)
                    update = res_approx.mul_(exp_avg)
                else:
                    update = exp_avg.clone()

                if group["weight_decay"] != 0:
                    p.data.add_(p.data, alpha=-group["weight_decay"] * group["lr"])

                update.mul_(group["lr"])
                p.data.add_(-update)

        return loss
