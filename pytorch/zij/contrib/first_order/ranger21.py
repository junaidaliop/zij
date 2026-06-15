# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa0)
# Copyright (c) Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the Ranger21 optimizer."""

import math
from typing import Callable, Optional

import torch
from torch.nn.functional import softplus

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["Ranger21"]


def _unit_norm(x: torch.Tensor, norm: float = 2.0) -> torch.Tensor:
    """Return the per-unit (row/slice-wise) norm of a tensor."""
    keep_dim = True
    dim: Optional[tuple[int, ...] | int] = None

    x_len = len(x.shape)
    if x_len <= 1:
        keep_dim = False
    elif x_len in (2, 3):
        dim = 1
    elif x_len == 4:
        dim = (1, 2, 3)
    else:
        dim = tuple(range(1, x_len))

    return x.norm(p=norm, dim=dim, keepdim=keep_dim)


def _agc(
    p: torch.Tensor,
    grad: torch.Tensor,
    agc_eps: float = 1e-3,
    agc_clip_val: float = 1e-2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Clip gradient values that exceed the unit-wise norm of the parameter."""
    max_norm = _unit_norm(p).clamp_min_(agc_eps).mul_(agc_clip_val)
    g_norm = _unit_norm(grad).clamp_min_(eps)

    clipped_grad = grad * (max_norm / g_norm)

    return torch.where(g_norm > max_norm, clipped_grad, grad)


def _centralize_gradient(grad: torch.Tensor, gc_conv_only: bool = False) -> None:
    """Subtract the per-unit gradient mean in place (gradient centralization)."""
    size = grad.dim()
    if (gc_conv_only and size > 3) or (not gc_conv_only and size > 1):
        grad.add_(-grad.mean(dim=tuple(range(1, size)), keepdim=True))


def _normalize_gradient(x: torch.Tensor, epsilon: float = 1e-8) -> None:
    """Normalize a gradient by its standard deviation in place."""
    if torch.numel(x) > 2:
        s = x.std().add_(epsilon)
        x.div_(s)


class Ranger21(Optimizer):
    r"""Implements Ranger21, a synergistic combination of AdamW and eight techniques.

    Ranger21 keeps an AdamW core and layers on adaptive gradient clipping,
    gradient centralization, gradient normalization, positive-negative
    momentum, norm loss, stable weight decay, a linear warmup combined with an
    explore-exploit warmdown schedule, Lookahead, and a softplus-smoothed
    denominator. The positive-negative momentum keeps two first-moment buffers,
    one for odd and one for even steps, and forms the update direction as a
    positively weighted current moment minus a negatively weighted previous
    moment, normalized so the learning rate need not change with ``beta0``:

    .. math::
       \begin{aligned}
            m_t &= \beta_1^2 m_{t-2} + (1 - \beta_1^2) g_t                       \\
            \hat{m}_t &= \frac{(1 + \beta_0) m_t - \beta_0 m_{t-1}}
                {\sqrt{(1 + \beta_2)^2 + \beta_2^2}}                              \\
            v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2,\quad
                v_t \leftarrow \max(v_t^{\max}, v_t)                             \\
            \theta_t &= \theta_{t-1} - \frac{\eta_t}{1 - \beta_1^t} \,
                \frac{\hat{m}_t}{\sqrt{v_t} / \sqrt{1 - \beta_2^t} + \epsilon}
       \end{aligned}

    The learning rate :math:`\eta_t` follows the explore-exploit schedule, a
    linear warmup over the first :math:`t_{\text{warmup}}` steps, a flat phase,
    and a linear warmdown over the last :math:`t_{\text{warmdown}}` steps, which
    is why ``num_iterations`` (the total number of training steps) is required.

    Note:
        Following the reference implementation, the positive-negative momentum
        combination fixes the coefficients to :math:`\beta_0 = 1` (so the update
        is :math:`2 m_t - m_{t-1}`) and normalizes by
        :math:`\sqrt{(1 + \beta_2)^2 + \beta_2^2}`; the ``beta0`` argument is
        retained only for the noise-amplitude validation range.

    Reference: Less Wright, Nestor Demeure,
    "Ranger21: a synergistic deep learning optimizer", arXiv 2021.
    https://arxiv.org/abs/2106.13731
    """

    def __init__(
        self,
        params: ParamsT,
        num_iterations: int,
        lr: float = 1e-3,
        beta0: float = 0.9,
        betas: tuple[float, float] = (0.9, 0.999),
        use_softplus: bool = True,
        beta_softplus: float = 50.0,
        disable_lr_scheduler: bool = False,
        num_warm_up_iterations: Optional[int] = None,
        num_warm_down_iterations: Optional[int] = None,
        warm_down_min_lr: float = 3e-5,
        agc_clipping_value: float = 1e-2,
        agc_eps: float = 1e-3,
        centralize_gradients: bool = True,
        normalize_gradients: bool = True,
        lookahead_merge_time: int = 5,
        lookahead_blending_alpha: float = 0.5,
        weight_decay: float = 1e-4,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        norm_loss_factor: float = 1e-4,
        eps: float = 1e-8,
        maximize: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if warm_down_min_lr < 0.0:
            raise ValueError(
                f"Invalid warm_down_min_lr: {warm_down_min_lr} - should be >= 0.0"
            )
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= beta0 <= 1.0:
            raise ValueError(f"Invalid beta0 parameter: {beta0} - should be in [0.0, 1.0]")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay} - should be >= 0.0")
        if agc_clipping_value < 0.0:
            raise ValueError(
                f"Invalid agc_clipping_value: {agc_clipping_value} - should be >= 0.0"
            )
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        if agc_eps < 0.0:
            raise ValueError(f"Invalid agc_eps: {agc_eps} - should be >= 0.0")

        self.min_lr = warm_down_min_lr
        self.use_softplus = use_softplus
        self.beta_softplus = beta_softplus
        self.disable_lr_scheduler = disable_lr_scheduler
        self.agc_clipping_value = agc_clipping_value
        self.agc_eps = agc_eps
        self.centralize_gradients = centralize_gradients
        self.normalize_gradients = normalize_gradients
        self.lookahead_merge_time = lookahead_merge_time
        self.lookahead_blending_alpha = lookahead_blending_alpha
        self.norm_loss_factor = norm_loss_factor
        self.maximize = maximize

        self.lookahead_step = 0
        self.starting_lr = lr
        self.current_lr = lr

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "weight_decouple": weight_decouple,
            "fixed_decay": fixed_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)

        self.num_warm_up_iterations = (
            self.build_warm_up_iterations(num_iterations, betas[1])
            if num_warm_up_iterations is None
            else num_warm_up_iterations
        )
        self.num_warm_down_iterations = (
            self.build_warm_down_iterations(num_iterations)
            if num_warm_down_iterations is None
            else num_warm_down_iterations
        )
        self.start_warm_down = num_iterations - self.num_warm_down_iterations
        self.warm_down_lr_delta = self.starting_lr - self.min_lr

    @staticmethod
    def build_warm_up_iterations(
        total_iterations: int, beta2: float, warm_up_pct: float = 0.22
    ) -> int:
        """Default un-tuned linear warmup length."""
        warm_up_iterations = math.ceil(2.0 / (1.0 - beta2))
        beta_pct = warm_up_iterations / total_iterations
        return (
            int(warm_up_pct * total_iterations)
            if beta_pct > 0.45
            else warm_up_iterations
        )

    @staticmethod
    def build_warm_down_iterations(
        total_iterations: int, warm_down_pct: float = 0.72
    ) -> int:
        """Default explore-exploit warmdown length."""
        start_warm_down = int(warm_down_pct * total_iterations)
        return total_iterations - start_warm_down

    def warm_up_dampening(self, lr: float, step: int) -> float:
        if step > self.num_warm_up_iterations:
            return lr

        warm_up_current_pct = min(1.0, (step / self.num_warm_up_iterations))
        self.current_lr = lr * warm_up_current_pct
        return self.current_lr

    def warm_down(self, lr: float, iteration: int) -> float:
        if iteration < self.start_warm_down:
            return lr

        warm_down_iteration = max((iteration + 1) - self.start_warm_down, 1)
        warm_down_pct = min(
            warm_down_iteration / (self.num_warm_down_iterations + 1), 1.0
        )

        self.current_lr = max(
            self.starting_lr - self.warm_down_lr_delta * warm_down_pct, self.min_lr
        )
        return self.current_lr

    def init_group(self, group: dict) -> None:
        if "step" not in group:
            group["step"] = 0

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("Ranger21 does not support sparse gradients")
            if torch.is_complex(p):
                raise RuntimeError("Ranger21 does not support complex parameters")

            state = self.state[p]
            if len(state) == 0:
                state["grad_ma"] = torch.zeros_like(p)
                state["variance_ma"] = torch.zeros_like(p)
                state["lookahead_params"] = p.clone()
                state["neg_grad_ma"] = torch.zeros_like(p)
                state["max_variance_ma"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        param_size = 0
        variance_ma_sum = 1.0

        for group in self.param_groups:
            self.init_group(group)
            group["step"] += 1

            _, beta2 = group["betas"]
            bias_correction2 = 1.0 - beta2 ** group["step"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                param_size += p.numel()

                if self.maximize:
                    grad.neg_()

                state = self.state[p]

                grad.copy_(_agc(p, grad, self.agc_eps, self.agc_clipping_value))

                _centralize_gradient(grad, gc_conv_only=False)
                _normalize_gradient(grad)

                variance_ma = state["variance_ma"]
                variance_ma.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                variance_ma_sum += (variance_ma / bias_correction2).sum()

        if param_size == 0:
            raise ValueError("Ranger21 received an empty parameter list")

        variance_normalized = math.sqrt(variance_ma_sum / param_size)

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            bias_correction1 = 1.0 - beta1 ** group["step"]
            bias_correction2_sq = math.sqrt(1.0 - beta2 ** group["step"])

            noise_norm = math.sqrt((1.0 + beta2) ** 2 + beta2**2)

            if self.disable_lr_scheduler:
                lr = group["lr"]
            else:
                lr = self.warm_up_dampening(group["lr"], group["step"])
                lr = self.warm_down(lr, group["step"])

            step_size = lr / bias_correction1

            for p in group["params"]:
                if p.grad is None:
                    continue

                if group["weight_decouple"]:
                    p.mul_(
                        1.0
                        - group["weight_decay"]
                        * (1.0 if group["fixed_decay"] else lr)
                        * (1.0 / variance_normalized)
                    )
                elif group["weight_decay"] > 0.0:
                    p.grad.add_(p, alpha=group["weight_decay"])

                correction = (
                    2.0
                    * self.norm_loss_factor
                    * (1.0 - 1.0 / _unit_norm(p).add_(group["eps"]))
                )
                p.mul_(1.0 - lr * correction)

                state = self.state[p]
                if group["step"] % 2 == 1:
                    grad_ma, neg_grad_ma = state["grad_ma"], state["neg_grad_ma"]
                else:
                    grad_ma, neg_grad_ma = state["neg_grad_ma"], state["grad_ma"]

                variance_ma = state["variance_ma"]
                torch.max(state["max_variance_ma"], variance_ma, out=variance_ma)

                de_nom = (variance_ma.sqrt() / bias_correction2_sq).add_(group["eps"])

                if self.use_softplus:
                    de_nom = softplus(de_nom, beta=self.beta_softplus)

                grad = p.grad
                _centralize_gradient(grad, gc_conv_only=False)
                _normalize_gradient(grad)

                grad_ma.mul_(beta1**2).add_(grad, alpha=1.0 - beta1**2)

                pn_momentum = (
                    grad_ma.mul(2.0)
                    .add_(neg_grad_ma, alpha=-1.0)
                    .mul_(1.0 / noise_norm)
                )
                p.addcdiv_(pn_momentum, de_nom, value=-step_size)

        self.lookahead_process_step()

        return loss

    def lookahead_process_step(self) -> None:
        self.lookahead_step += 1
        if self.lookahead_step >= self.lookahead_merge_time:
            self.lookahead_step = 0
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    p.mul_(self.lookahead_blending_alpha).add_(
                        state["lookahead_params"],
                        alpha=1.0 - self.lookahead_blending_alpha,
                    )
                    state["lookahead_params"].copy_(p)
