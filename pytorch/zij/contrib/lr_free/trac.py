# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa0)
# Copyright (c) Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the TRAC parameter-free optimizer wrapper."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from ...core.optimizer import Optimizer

__all__ = ["TRAC"]


def polyval(x: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
    r"""Evaluate a polynomial at ``x`` using Horner's scheme.

    Args:
        x: variable at which to evaluate the polynomial.
        coef: coefficients ordered from highest degree to lowest.
    """
    result = coef[0].clone()

    for c in coef[1:]:
        result = (result * x) + c

    return result[0]


class ERF1994(nn.Module):
    r"""Polynomial approximation of the complex error function (ERF1994).

    Args:
        num_coefs: number of polynomial coefficients used in the approximation.
    """

    def __init__(self, num_coefs: int = 128) -> None:
        super().__init__()

        self.n: int = num_coefs

        self.i: torch.Tensor = torch.complex(torch.tensor(0.0), torch.tensor(1.0))
        self.m = 2 * self.n
        self.m2 = 2 * self.m
        self.k = torch.linspace(-self.m + 1, self.m - 1, self.m2 - 1)
        self.l = torch.sqrt(self.n / torch.sqrt(torch.tensor(2.0)))
        self.theta = self.k * torch.pi / self.m
        self.t = self.l * torch.tan(self.theta / 2.0)
        self.f = torch.exp(-self.t ** 2) * (self.l ** 2 + self.t ** 2)  # fmt: skip
        self.a = torch.fft.fft(torch.fft.fftshift(self.f)).real / self.m2
        self.a = torch.flipud(self.a[1:self.n + 1])  # fmt: skip

    def w_algorithm(self, z: torch.Tensor) -> torch.Tensor:
        r"""Compute the Faddeeva function of a complex number.

        Args:
            z: a tensor of complex numbers.
        """
        self.l = self.l.to(z.device)
        self.i = self.i.to(z.device)
        self.a = self.a.to(z.device)

        iz = self.i * z
        lp_iz, ln_iz = self.l + iz, self.l - iz

        z_ = lp_iz / ln_iz
        p = polyval(z_.unsqueeze(0), self.a)
        return 2 * p / ln_iz.pow(2) + (1.0 / torch.sqrt(torch.tensor(torch.pi))) / ln_iz

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        r"""Compute the error function of a complex number.

        Args:
            z: a tensor of complex numbers.
        """
        sign_r = torch.sign(z.real)
        sign_i = torch.sign(z.imag)
        z = torch.complex(torch.abs(z.real), torch.abs(z.imag))
        out = -torch.exp(torch.log(self.w_algorithm(z * self.i)) - z ** 2) + 1  # fmt: skip
        return torch.complex(out.real * sign_r, out.imag * sign_i)


class TRAC:
    r"""Implements TRAC, a parameter-free scale tuner for any base optimizer.

    TRAC keeps the reference point :math:`\theta_{ref}` (the parameters before
    optimization began) and, after each base-optimizer update, rescales the
    cumulative displacement by a learned scale :math:`S_t`. To recover the base
    optimizer's raw step direction, the displacement is first un-scaled by the
    previous-step scale :math:`S_{t-1}`, giving the un-scaled displacement
    :math:`\Delta_t = (\theta_t - \theta_{ref}) / (S_{t-1} + \epsilon)`. The scale
    is the sum of :math:`n` one-dimensional discounted tuners, one per discount
    factor :math:`\beta_i`. With base update producing :math:`\theta_t`, gradient
    :math:`g_t`, and inner product :math:`h_t`:

    .. math::
       \Delta_t &= \frac{\theta_t - \theta_{ref}}{S_{t-1} + \epsilon} \\
       h_t &= \langle g_t,\, \Delta_t \rangle \\
       v_{t,i} &= \beta_i^{2}\, v_{t-1,i} + h_t^{2} \\
       \sigma_{t,i} &= \beta_i\, \sigma_{t-1,i} - h_t \\
       s_{t,i} &= \frac{s_{init}}{\mathrm{erfi}(1/\sqrt{2})}\,
           \mathrm{erfi}\!\Bigl(\frac{\sigma_{t,i}}{\sqrt{2 v_{t,i}} + \epsilon}\Bigr) \\
       S_t &= \max\Bigl(0,\, \sum_{i=1}^{n} s_{t,i}\Bigr) \\
       \theta_{t+1} &= \theta_{ref} + S_t\,\Delta_t

    where :math:`\mathrm{erfi}` is the imaginary error function and
    :math:`s_{init}` is the initial scale ``s_prev``.

    Reference: Aneesh Muppidi, Zhiyu Zhang, Heng Yang,
    "Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning",
    NeurIPS 2024.
    https://arxiv.org/abs/2405.16642

    Note: this is a wrapper around a base optimizer. Pass an already constructed
    optimizer instance, e.g.
    ``TRAC(torch.optim.AdamW(model.parameters()))``.
    """

    def __init__(
        self,
        optimizer: Union[Optimizer, type],
        betas: List[float] = (0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999),
        num_coefs: int = 128,
        s_prev: float = 1e-8,
        eps: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        if num_coefs <= 0:
            raise ValueError("num_coefs must be positive")
        if s_prev < 0.0:
            raise ValueError("s_prev must be non-negative")
        if eps < 0.0:
            raise ValueError("eps must be non-negative")

        self._optimizer_step_pre_hooks: Dict[int, Callable] = {}
        self._optimizer_step_post_hooks: Dict[int, Callable] = {}

        self.optimizer: Optimizer = self.load_optimizer(optimizer, **kwargs)

        self.betas = betas
        self.s_prev = s_prev
        self.eps = eps

        self.erf: nn.Module = ERF1994(num_coefs=num_coefs)
        self.f_term: torch.Tensor = self.s_prev / self.erf_imag(
            1.0 / torch.sqrt(torch.tensor(2.0))
        )

        self.defaults: Dict = self.optimizer.defaults

    def __str__(self) -> str:
        return "TRAC"

    @staticmethod
    def load_optimizer(optimizer: Union[Optimizer, type], **kwargs: Any) -> Optimizer:
        """Return a base optimizer instance, building it from a class if needed."""
        if not isinstance(optimizer, type):
            return optimizer

        if "params" in kwargs:
            params = kwargs.pop("params")
            return optimizer(params, **kwargs)

        raise ValueError("need to pass `params` when you pass the optimizer class.")

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self) -> Dict:
        return self.optimizer.state

    def state_dict(self) -> Dict:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict) -> None:
        self.optimizer.load_state_dict(state_dict)

    def init_group(self, group: Dict, **kwargs: Any) -> None:
        if "step" not in group:
            group["step"] = 0

        updates: Dict[torch.Tensor, torch.Tensor] = kwargs.get("updates", {})

        for p in group["params"]:
            self.state["trac"][p] = updates[p].clone()

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def erf_imag(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(x):
            x = x.real.to(torch.float32)

        ix = torch.complex(torch.zeros_like(x), x)

        return self.erf(ix).imag

    @torch.no_grad()
    def backup_params_and_grads(self) -> Tuple[Dict, Dict]:
        updates, grads = {}, {}

        for group in self.param_groups:
            for p in group["params"]:
                updates[p] = p.clone()
                grads[p] = p.grad.clone() if p.grad is not None else None

        return updates, grads

    @torch.no_grad()
    def trac_step(self, updates: Dict, grads: Dict) -> None:
        self.state["trac"]["step"] += 1

        deltas = {}

        device = self.param_groups[0]["params"][0].device

        s = self.state["trac"]["s"]
        h = torch.zeros((1,), device=device)
        for group in self.param_groups:
            for p in group["params"]:
                if grads[p] is None:
                    continue

                theta_ref = self.state["trac"][p]
                update = updates[p]

                deltas[p] = (update - theta_ref) / s.add(self.eps)
                update.neg_().add_(p)

                grad, delta = grads[p], deltas[p]

                product = torch.dot(delta.flatten(), grad.flatten())
                h.add_(product)

                delta.add_(update)

                p.copy_(theta_ref)

        betas = self.state["trac"]["betas"]
        variance = self.state["trac"]["variance"]
        sigma = self.state["trac"]["sigma"]

        variance.mul_(betas.pow(2)).add_(h.pow(2))
        sigma.mul_(betas).sub_(h)

        term = self.erf_imag(sigma / (2.0 * variance).sqrt_().add_(self.eps)).mul_(
            self.f_term
        )
        s.copy_(torch.sum(term))

        scale = max(s, 0.0)

        for group in self.param_groups:
            for p in group["params"]:
                if grads[p] is None:
                    continue

                p.add_(deltas[p] * scale)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        with torch.enable_grad():
            loss = self.optimizer.step(closure)

        updates, grads = self.backup_params_and_grads()

        if "trac" not in self.state:
            device = self.param_groups[0]["params"][0].device

            self.state["trac"] = {
                "betas": torch.tensor(self.betas, device=device),
                "s": torch.zeros(1, device=device),
                "variance": torch.zeros(len(self.betas), device=device),
                "sigma": torch.full((len(self.betas),), 1e-8, device=device),
                "step": 0,
            }

            for group in self.param_groups:
                self.init_group(group, updates=updates)

        self.trac_step(updates, grads)

        return loss
