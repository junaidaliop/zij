# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa0)
# Copyright (c) Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
"""Implementation of the AdaMuon optimizer."""

import math
from typing import List, Tuple, Union

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["AdaMuon"]

NewtonSchulzWeight = Tuple[float, float, float]
NewtonSchulzWeights = Union[
    str, NewtonSchulzWeight, List[NewtonSchulzWeight], Tuple[NewtonSchulzWeight, ...]
]

NS_COEFFICIENTS = {
    "original": [
        # Keller Jordan's Muon.
        (3.4445, -4.7750, 2.0315),
    ],
    "quintic": [
        # Optimized quintic schedules from modded-nanogpt.
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ],
    "polar_express": [
        # Polar Express schedule with safety 1e-2.
        (8.237312490495555, -23.157747414558198, 16.680568411445915),
        (4.082441999064835, -2.893047735332586, 0.5252849256975648),
        (3.9263479922546582, -2.8547468034765298, 0.5318022422894988),
        (3.2982187133085143, -2.424541981026706, 0.48632008358844075),
        (2.2970369434552573, -1.63662558125903, 0.4002628455953627),
        (1.8763805351440397, -1.2347896577722228, 0.35891887501668385),
        (1.8564423485617974, -1.2132449880935525, 0.3568003487825883),
        (1.8749994008682747, -1.2499988017229169, 0.3749994008546422),
    ],
    "polar_express_safer": [
        # Polar Express safer schedule with safety 2e-2.
        (8.156554524902461, -22.48329292557795, 15.878769915207462),
        (4.0429299351667245, -2.808917465908704, 0.5000178451051299),
        (3.8916678022926563, -2.7724841532176825, 0.5060648178503389),
        (3.285753657755658, -2.3681294933425394, 0.46449024233003117),
        (2.3005307116270983, -1.6111665557258408, 0.3833374427545273),
        (1.8631210546382593, -1.2042160621002727, 0.3421879560523383),
        (1.8382572152247512, -1.1779263289537742, 0.3396513038637379),
        (1.8749999923301852, -1.2499999836060613, 0.374999991275876),
    ],
}


def get_newton_schulz_weights(weights: NewtonSchulzWeights) -> List[NewtonSchulzWeight]:
    """Resolve Newton-Schulz quintic coefficients from a preset name or explicit
    coefficients."""
    if isinstance(weights, str):
        key = weights.lower()
        if key in NS_COEFFICIENTS:
            return list(NS_COEFFICIENTS[key])
        raise ValueError(
            f"Invalid `weights` string choice. expected one of {tuple(NS_COEFFICIENTS.keys())}."
        )

    if isinstance(weights, tuple) and len(weights) == 3:
        w0, w1, w2 = weights
        if all(isinstance(w, (int, float)) for w in (w0, w1, w2)):
            return [(float(w0), float(w1), float(w2))]

    if len(weights) == 0:
        raise ValueError("`weights` schedule must not be empty.")

    normalized: List[NewtonSchulzWeight] = []
    for coeff in weights:
        if not isinstance(coeff, tuple) or len(coeff) != 3:
            raise ValueError(
                "`weights` must be a preset name, a coefficient tuple, or a list of coefficient tuples."
            )
        c0, c1, c2 = coeff
        normalized.append((float(c0), float(c1), float(c2)))

    return normalized


def zero_power_via_newton_schulz_5(
    g: torch.Tensor,
    num_steps: int = 5,
    eps: float = 1e-7,
    safety_factor: float = 1.0,
    weights: NewtonSchulzWeights = (3.4445, -4.7750, 2.0315),
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    r"""Compute the zeroth power / orthogonalization of G via the Newton-Schulz
    quintic iteration.

    The iteration uses coefficients selected to maximize the slope at zero. It
    does not produce :math:`UV^\top` exactly but rather :math:`US'V^\top` with
    :math:`S'` diagonal whose entries are roughly uniform in ``[0.5, 1.5]``,
    which does not hurt model performance relative to :math:`UV^\top` from the
    singular value decomposition :math:`USV^\top = G`.
    """
    if g.ndim < 2:
        raise ValueError(f"input must be over 2-dimensional. got {g.ndim}D.")

    weight_schedule = get_newton_schulz_weights(weights)
    coeff_sequence = [
        weight_schedule[min(i, len(weight_schedule) - 1)] for i in range(num_steps)
    ]

    x = g.to(dtype=dtype, copy=True)

    transpose: bool = x.size(-2) > x.size(-1)
    if transpose:
        x = x.mT

    x.div_(x.norm(2, dim=(-2, -1), keepdim=True).mul_(safety_factor).clamp_min_(eps))

    mm_fn = torch.baddbmm if x.ndim > 2 else torch.addmm

    x = x.contiguous()
    a = torch.empty((*x.shape[:-1], x.size(-2)), device=x.device, dtype=x.dtype)
    b = torch.empty_like(a)
    c = torch.empty_like(x)

    for w0, w1, w2 in coeff_sequence:
        mm_fn(a, x, x.mT, beta=0.0, alpha=1.0, out=a)
        mm_fn(a, a, a, beta=w1, alpha=w2, out=b)
        mm_fn(x, b, x, beta=w0, alpha=1.0, out=c)
        x, c = c, x

    return x.mT if transpose else x


def get_adjusted_lr(
    lr: float, param_shape: Tuple[float, ...], use_adjusted_lr: bool = False
) -> float:
    r"""Adjust the learning rate to match the update RMS across rectangular
    matrices."""
    output_shape, *input_shape = param_shape
    input_shape = math.prod(input_shape)

    ratio: float = (
        math.pow(max(1.0, output_shape / input_shape), 0.5)
        if use_adjusted_lr
        else 0.2 * math.sqrt(max(output_shape, input_shape))
    )

    return lr * ratio


class AdaMuon(Optimizer):
    r"""Implements AdaMuon, an adaptive variance-normalized Muon optimizer.

    AdaMuon augments Muon with an element-wise second moment applied to the
    orthogonalized update. For a matrix parameter, the momentum :math:`M_t` is
    orthogonalized through a Newton-Schulz iteration, a per-element second
    moment :math:`V_t` is accumulated on the orthogonalized direction, and the
    direction is variance-normalized before an RMS-aligned rescaling that
    matches the update magnitude to Adam:

    .. math::
       \begin{aligned}
       M_t &= \beta_1 M_{t-1} + (1 - \beta_1) G_t \\
       O_t &= \mathrm{NewtonSchulz}(M_t) \\
       V_t &= \beta_2 V_{t-1} + (1 - \beta_2)\, O_t \odot O_t \\
       \hat{O}_t &= O_t \oslash \left(\sqrt{V_t / (1 - \beta_2^t)} + \epsilon\right) \\
       \theta_t &= \theta_{t-1} - \gamma\,
           \frac{0.2 \sqrt{mn}}{\lVert \hat{O}_t \rVert_F}\, \hat{O}_t
       \end{aligned}

    where :math:`m, n` are the matrix dimensions and :math:`\odot`, :math:`\oslash`
    denote element-wise product and division. Parameters in a group with
    ``use_muon=False`` are updated with decoupled-weight-decay AdamW instead, so
    embeddings, heads, and scalar or vector parameters can share the optimizer.

    This implementation follows kozistr/pytorch_optimizer and omits the paper's
    :math:`\mathrm{Sign}(M_t)` sign-stabilization step before Newton-Schulz; that
    is, it computes :math:`O_t = \mathrm{NewtonSchulz}(M_t)` rather than the
    paper's :math:`O_t = \mathrm{NewtonSchulz}(\mathrm{Sign}(M_t))`.

    Unlike the paper, which applies no bias correction on :math:`V_t` (the
    RMS-alignment rescale removes it), this implementation (following kozistr)
    applies second-moment bias correction via :math:`1 - \beta_2^t`. This factor
    is cancelled by the subsequent RMS rescale, so the resulting update is
    numerically unchanged.

    Reference: Chongjie Si, Debing Zhang, Wei Shen, "AdaMuon: Adaptive Muon
    Optimizer", 2025.
    https://arxiv.org/abs/2507.11005
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 2e-2,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        ns_steps: int = 5,
        ns_coeffs: NewtonSchulzWeights = "original",
        use_adjusted_lr: bool = False,
        adamw_lr: float = 3e-4,
        adamw_betas: Tuple[float, float] = (0.9, 0.999),
        adamw_wd: float = 0.0,
        eps: float = 1e-10,
        maximize: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Learning rate {lr} must be non-negative")
        if adamw_lr < 0.0:
            raise ValueError(f"adamw_lr {adamw_lr} must be non-negative")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay {weight_decay} must be non-negative")
        if ns_steps <= 0:
            raise ValueError(f"ns_steps {ns_steps} must be positive")
        if adamw_wd < 0.0:
            raise ValueError(f"adamw_wd {adamw_wd} must be non-negative")
        if eps < 0.0:
            raise ValueError(f"eps {eps} must be non-negative")
        for name, pair in (("betas", betas), ("adamw_betas", adamw_betas)):
            if not 0.0 <= pair[0] < 1.0:
                raise ValueError(f"{name}[0] {pair[0]} must be in the range [0, 1)")
            if not 0.0 <= pair[1] < 1.0:
                raise ValueError(f"{name}[1] {pair[1]} must be in the range [0, 1)")
        ns_coeffs = get_newton_schulz_weights(ns_coeffs)

        self.maximize = maximize

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups, "use_muon": True}]

        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("`use_muon` must be set.")

            if group["use_muon"]:
                group.setdefault("lr", lr)
                group.setdefault("betas", betas)
                group.setdefault("weight_decay", weight_decay)
                group.setdefault("ns_steps", ns_steps)
                group["ns_coeffs"] = get_newton_schulz_weights(
                    group.get("ns_coeffs", ns_coeffs)
                )
                group.setdefault("use_adjusted_lr", use_adjusted_lr)
            else:
                group.setdefault("lr", adamw_lr)
                group.setdefault("betas", adamw_betas)
                group.setdefault("weight_decay", adamw_wd)

            group.setdefault("weight_decouple", weight_decouple)
            group.setdefault("eps", eps)

        super().__init__(param_groups, {})

    def init_group(self, group):
        if "step" not in group:
            group["step"] = 0

        for p in group["params"]:
            if p.grad is None:
                continue

            if p.grad.is_sparse:
                raise RuntimeError("AdaMuon does not support sparse gradients")
            if torch.is_complex(p):
                raise RuntimeError("AdaMuon does not support complex parameters")

            state = self.state[p]
            if len(state) == 0:
                if group["use_muon"]:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p.flatten())
                else:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group["step"] += 1

            beta1, beta2 = group["betas"]

            bias_correction1 = 1.0 - beta1 ** group["step"]
            bias_correction2 = 1.0 - beta2 ** group["step"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if self.maximize:
                    grad.neg_()

                state = self.state[p]

                if group["weight_decouple"]:
                    p.mul_(1.0 - group["weight_decay"] * group["lr"])
                elif group["weight_decay"] > 0.0:
                    grad.add_(p, alpha=group["weight_decay"])

                if group["use_muon"]:
                    m = state["m"]
                    m.lerp_(grad, weight=1.0 - beta1)

                    update = m.clone()

                    if update.ndim > 2:
                        update = update.view(len(update), -1)

                    update = zero_power_via_newton_schulz_5(
                        update, num_steps=group["ns_steps"], weights=group["ns_coeffs"]
                    ).flatten()

                    v = state["v"]
                    v.mul_(beta2).addcmul_(update, update, value=1.0 - beta2)

                    update.div_((v / bias_correction2).sqrt_().add_(group["eps"]))
                    update = update.reshape(p.size())

                    update.mul_(0.2 * math.sqrt(p.numel())).div_(
                        update.norm().add_(group["eps"])
                    )

                    lr = get_adjusted_lr(
                        group["lr"], p.size(), use_adjusted_lr=group["use_adjusted_lr"]
                    )

                    p.add_(update, alpha=-lr)
                else:
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                    exp_avg.lerp_(grad, weight=1.0 - beta1)
                    exp_avg_sq.lerp_(grad.square(), weight=1.0 - beta2)

                    de_nom = (
                        exp_avg_sq.sqrt()
                        .add_(group["eps"])
                        .div_(math.sqrt(bias_correction2))
                    )

                    p.addcdiv_(exp_avg / bias_correction1, de_nom, value=-group["lr"])

        return loss
