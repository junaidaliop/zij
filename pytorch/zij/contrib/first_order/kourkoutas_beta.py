# Adapted from https://github.com/sck-at-ucy/kbeta (commit 7ae85e2)
# Copyright (c) 2025 Stavros Kassinos. Licensed under the MIT license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the Kourkoutas-beta optimizer."""

from typing import Any, Callable, Optional

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["KourkoutasSoftmaxFlex"]


class KourkoutasSoftmaxFlex(Optimizer):
    r"""Implements Kourkoutas-beta, an Adam variant with a layer-wise dynamic
    :math:`\beta_2` driven by a bounded "sunspike" ratio.

    For each layer the optimizer tracks an exponential moving average of the
    pooled gradient norm and compares the current norm against it. A large
    ratio (a gradient spike) lowers :math:`\beta_2` toward :math:`\beta_{2,\min}`
    so the second moment reacts faster; a calm phase keeps :math:`\beta_2` near
    :math:`\beta_{2,\max}`, recovering Adam-like behavior.

    .. math::
       \begin{aligned}
            n_t &= \lVert g_t \rVert_2                                            \\
            e_t &= \alpha\, e_{t-1} + (1 - \alpha)\, n_t                          \\
            r_t &= \frac{n_t}{e_t + \tau}                                         \\
            s_t &= \frac{r_t}{1 + r_t}                                            \\
            \beta_{2,t} &= \beta_{2,\max}
                - (\beta_{2,\max} - \beta_{2,\min})\, s_t                         \\
            m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t                            \\
            v_t &= \beta_{2,t} v_{t-1} + (1 - \beta_{2,t}) g_t^2                  \\
            \theta_t &= \theta_{t-1} - \eta\,
                \frac{m_t}{\sqrt{v_t} + \epsilon}
       \end{aligned}

    The norm :math:`n_t` is pooled over every parameter in a layer, so the
    sunspike :math:`s_t \in [0, 1)` and the resulting :math:`\beta_{2,t}` are
    shared by all tensors of that layer. During the first ``warmup_steps`` the
    sunspike is held at zero and :math:`\beta_2` is fixed at the midpoint
    :math:`\tfrac{1}{2}(\beta_{2,\min} + \beta_{2,\max})`. The constant
    :math:`\tau` is ``tiny_spike``.

    Optional features (all off by default) are leaky-AMSGrad on the second
    moment (``decay``), a trust-region clip :math:`\lvert \Delta\theta \rvert
    \le \eta \cdot \mathrm{max\_ratio}` (``max_ratio``), an adaptive tiny term
    that scales the denominator floor with :math:`\langle \lvert\theta\rvert
    \rangle` (``adaptive_tiny``), and bias correction (``bias_correction``).
    With all features off, ``bias_correction="none"``, and
    :math:`\beta_{2,\min} = \beta_{2,\max}`, the method reduces to Adam.

    Note:
        Each parameter group is treated as one layer: the sunspike ratio and
        :math:`\beta_2` are pooled across the group's parameters. Split the
        parameters into separate groups to obtain finer-grained layer-wise
        :math:`\beta_2`.

    Reference: Stavros C. Kassinos, "Kourkoutas-Beta: A Sunspike-Driven Adam
    Optimizer with Desert Flair", 2025.
    https://arxiv.org/abs/2508.12996
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2_max: float = 0.999,
        beta2_min: float = 0.9,
        eps: float = 1e-8,
        alpha: float = 0.9,
        tiny_spike: float = 1e-8,
        tiny_denom: float = 1e-8,
        decay: Optional[float] = None,
        max_ratio: Optional[float] = None,
        adaptive_tiny: bool = False,
        bias_correction: str = "none",
        warmup_steps: int = 0,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 value: {beta1}")
        if not 0.0 <= beta2_min <= beta2_max < 1.0:
            raise ValueError(
                f"Invalid beta2 range: ({beta2_min}, {beta2_max})"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if decay is not None and not 0.0 <= decay <= 1.0:
            raise ValueError(f"Invalid decay value: {decay}")
        if max_ratio is not None and not 0.0 < max_ratio:
            raise ValueError(f"Invalid max_ratio value: {max_ratio}")
        if bias_correction not in {"none", "beta2max", "exact"}:
            raise ValueError(f"Invalid bias_correction: {bias_correction}")
        if warmup_steps < 0:
            raise ValueError(f"Invalid warmup_steps value: {warmup_steps}")

        defaults = {
            "lr": lr,
            "beta1": beta1,
            "beta2_max": beta2_max,
            "beta2_min": beta2_min,
            "eps": eps,
            "alpha": alpha,
            "tiny_spike": tiny_spike,
            "tiny_denom": tiny_denom,
            "decay": decay,
            "max_ratio": max_ratio,
            "adaptive_tiny": adaptive_tiny,
            "bias_correction": bias_correction,
            "warmup_steps": warmup_steps,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("decay", None)
            group.setdefault("max_ratio", None)
            group.setdefault("adaptive_tiny", False)
            group.setdefault("bias_correction", "none")
            group.setdefault("warmup_steps", 0)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if "step" not in group:
                group["step"] = 0
                group["grad_ema"] = 0.0
            group["step"] += 1
            step = group["step"]

            beta1 = group["beta1"]
            beta2_max = group["beta2_max"]
            beta2_min = group["beta2_min"]
            eps = group["eps"]
            alpha = group["alpha"]
            decay = group["decay"]
            max_ratio = group["max_ratio"]
            adaptive_tiny = group["adaptive_tiny"]
            bias_correction = group["bias_correction"]
            lr = group["lr"]
            use_v_max = (decay is not None) or (max_ratio is not None)

            grads = []
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "Kourkoutas-beta does not support sparse gradients"
                    )
                grads.append((p, p.grad))

            if not grads:
                continue

            # Pooled gradient norm and the layer-wide dynamic beta2.
            sum_sq = sum(g.pow(2).sum() for _, g in grads)
            g_norm = sum_sq.sqrt().item()

            group["grad_ema"] = alpha * group["grad_ema"] + (1.0 - alpha) * g_norm
            g_ema = group["grad_ema"]

            if step <= group["warmup_steps"]:
                sun = 0.0
                beta2 = 0.5 * (beta2_min + beta2_max)
            else:
                raw = g_norm / (g_ema + group["tiny_spike"])
                sun = raw / (1.0 + raw)
                beta2 = beta2_max - (beta2_max - beta2_min) * sun

            for p, grad in grads:
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["beta2_cumprod"] = 1.0
                    if use_v_max:
                        state["v_max"] = torch.zeros_like(p)

                m, v = state["m"], state["v"]
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if use_v_max:
                    v_max = state["v_max"]
                    if decay is not None:
                        v_max.mul_(decay)
                    torch.maximum(v_max, v, out=v_max)
                    v_hat = v_max
                else:
                    v_hat = v

                if adaptive_tiny:
                    tiny_local = group["tiny_denom"] * max(p.abs().mean().item(), 1.0)
                else:
                    tiny_local = 0.0

                if bias_correction == "none":
                    denom = v_hat.sqrt().add_(tiny_local + eps)
                    upd = denom.reciprocal_().mul_(m).mul_(lr)
                else:
                    state["beta2_cumprod"] *= beta2
                    bc1 = 1.0 - beta1**step
                    if bias_correction == "exact":
                        bc2 = 1.0 - state["beta2_cumprod"]
                    else:
                        bc2 = 1.0 - beta2_max**step
                    denom = v_hat.div(bc2).sqrt_().add_(tiny_local + eps)
                    upd = denom.reciprocal_().mul_(m).mul_(lr / bc1)

                if max_ratio is not None:
                    lim = lr * max_ratio
                    upd.clamp_(-lim, lim)

                p.sub_(upd)

        return loss
