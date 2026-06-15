# Adapted from https://github.com/LIONS-EPFL/scion (commit f58a393)
# Copyright (c) 2024 LIONS. Licensed under the MIT license.
# See THIRD_PARTY_NOTICES.md for details.
"""Implementation of the Scion optimizer."""

import math

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["Scion"]


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    r"""Compute the zeroth power / orthogonalization of ``g`` via the quintic
    Newton-Schulz iteration.

    The coefficients are selected to maximize the slope at zero. The iteration
    does not produce :math:`UV^\top` exactly but rather :math:`US'V^\top` with
    :math:`S'` diagonal whose entries are roughly uniform in ``[0.5, 1.5]``,
    which does not hurt model performance relative to :math:`UV^\top` from the
    singular value decomposition :math:`USV^\top = g`.
    """
    if g.ndim != 2:
        raise ValueError(f"input must be 2-dimensional. got {g.ndim}D.")

    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.bfloat16()
    transpose = g.size(0) > g.size(1)
    if transpose:
        x = x.T

    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        s = x @ x.T
        t = b * s + c * s @ s
        x = a * x + t @ x

    if transpose:
        x = x.T
    return x


class Norm:
    """Base class for the linear-minimization-oracle norm backends."""

    def lmo(self, g):
        raise NotImplementedError

    def init(self, w):
        raise NotImplementedError


class ColNorm(Norm):
    """Column-wise normalization.

    Args:
        normalized: If ``True``, normalizes by the input dimension. Use ``True``
            only for non-input layers.
        transpose: If ``True``, transposes the input before normalization. Use
            ``True`` for embedding layers that store weights as
            ``(vocab_size, embedding_dim)``.
    """

    def __init__(self, normalized=False, transpose=False):
        self.normalized = normalized
        self.transpose = transpose

    def lmo(self, g):
        eps = 1e-8
        if self.transpose:
            g = g.transpose(0, 1)
        rms_values = (
            1 / math.sqrt(g.size(0)) * torch.sqrt(torch.sum(g ** 2, dim=0, keepdim=True))
        )
        if self.normalized:
            rms_values *= g.size(1)
        g = g / (rms_values + eps)
        if self.transpose:
            g = g.transpose(0, 1)
        return g

    def init(self, w):
        dtype = w.data.dtype
        if self.transpose:
            w.data = w.data.transpose(0, 1)
        torch.nn.init.normal_(w.data)
        w.data /= w.norm(dim=0, keepdim=True)
        w.data *= math.sqrt(w.size(0))
        if self.normalized:
            w.data /= w.size(1)
        w.data = w.data.to(dtype=dtype)
        if self.transpose:
            w.data = w.data.transpose(0, 1)
        return w


class RowNorm(Norm):
    """Row-wise normalization.

    Args:
        normalized: If ``True``, normalizes by the input dimension. Use ``False``
            only for the input layer.
        transpose: If ``True``, transposes the input before normalization. Use
            ``True`` for embedding layers that store weights as
            ``(vocab_size, embedding_dim)``.
    """

    def __init__(self, normalized=True, transpose=False):
        self.normalized = normalized
        self.transpose = transpose

    def lmo(self, g):
        eps = 1e-8
        if self.transpose:
            g = g.transpose(0, 1)
        rms_values = torch.sqrt(torch.sum(g ** 2, dim=-1, keepdim=True))
        if self.normalized:
            rms_values *= math.sqrt(g.size(-1))
        g = g / (rms_values + eps)
        if self.transpose:
            g = g.transpose(0, 1)
        return g

    def init(self, w):
        dtype = w.data.dtype
        if self.transpose:
            w.data = w.data.transpose(0, 1)
        torch.nn.init.normal_(w.data)
        w.data /= w.norm(dim=-1, keepdim=True)
        if self.normalized:
            w.data /= math.sqrt(w.size(-1))
        w.data = w.data.to(dtype=dtype)
        if self.transpose:
            w.data = w.data.transpose(0, 1)
        return w


class BiasRMS(Norm):
    """Root-mean-square normalization for bias and vector parameters."""

    def lmo(self, g):
        eps = 1e-8
        rms_values = torch.sqrt(torch.mean(g ** 2, dim=0, keepdim=True))
        g = g / (rms_values + eps)
        return g

    def init(self, g):
        return torch.nn.init.zeros_(g)


class SpectralConv(Norm):
    """Spectral normalization for convolutional weights."""

    def __init__(self, steps=5):
        self.steps = steps

    def lmo(self, g):
        g = zeropower_via_newtonschulz5(g.reshape(len(g), -1), steps=self.steps).view(
            g.shape
        )
        if g.ndim == 3:  # Conv1d
            out_channels, in_channels, k = g.shape
            g *= (out_channels / in_channels) ** 0.5 / k
        elif g.ndim == 4:  # Conv2d
            out_channels, in_channels, k, _ = g.shape
            g *= (out_channels / in_channels) ** 0.5 / (k ** 2)
        return g

    def init(self, w):
        w_fp = w.data.double()
        k = w.data.size(2)
        for kx in range(k):
            for ky in range(k):
                torch.nn.init.orthogonal_(w_fp[:, :, kx, ky])

        if w.ndim == 3:  # Conv1d
            out_channels, in_channels, k = w_fp.shape
            w_fp.mul_((out_channels / in_channels) ** 0.5 / k)
        elif w.ndim == 4:  # Conv2d
            out_channels, in_channels, k, _ = w_fp.shape
            w_fp.mul_((out_channels / in_channels) ** 0.5 / (k ** 2))
        w.data = w_fp.to(dtype=w.data.dtype)
        return w


class Spectral(Norm):
    """Spectral normalization for linear-layer weights."""

    def __init__(self, max=False, normalized=True, steps=5):
        self.max = max
        self.steps = steps
        self.normalized = normalized

    def lmo(self, g):
        g = zeropower_via_newtonschulz5(g.reshape(len(g), -1), steps=self.steps).view(
            g.shape
        )
        d_out, d_in = g.shape

        if self.normalized:
            scale = (d_out / d_in) ** 0.5
        else:
            scale = d_out ** 0.5
        if self.max:
            scale = max(1, scale)
        g *= scale

        return g

    def init(self, w):
        w_fp = w.data.double()
        torch.nn.init.orthogonal_(w_fp)
        d_out, d_in = w_fp.shape

        if self.normalized:
            scale = (d_out / d_in) ** 0.5
        else:
            scale = d_out ** 0.5
        if self.max:
            scale = max(1, scale)
        w_fp.mul_(scale)

        w.data = w_fp.to(dtype=w.data.dtype)
        return w


class Sign(Norm):
    """Sign normalization, the LMO over the entrywise infinity norm ball."""

    def __init__(self, zero_init=False, normalized=True):
        self.zero_init = zero_init
        self.normalized = normalized

    def lmo(self, g):
        d_out, d_in = g.shape
        if self.normalized:
            return (1 / d_in) * torch.sign(g)
        return torch.sign(g)

    def init(self, w):
        if self.zero_init:
            torch.nn.init.zeros_(w)
        else:
            # Generate -1/fan_in or 1/fan_in uniformly at random.
            d_out, d_in = w.shape
            w.data = torch.randint(0, 2, w.shape, dtype=w.dtype, device=w.device) * 2 - 1
            if self.normalized:
                w.data *= 1 / d_in
        return w


class Auto(Norm):
    """Dispatch to the appropriate norm backend based on parameter rank."""

    def lmo(self, g):
        if g.ndim in (3, 4):
            return SpectralConv().lmo(g)
        if g.ndim == 2:
            return Spectral().lmo(g)
        if g.ndim in (0, 1):
            return BiasRMS().lmo(g)

    def init(self, w):
        if w.ndim in (3, 4):
            return SpectralConv().init(w)
        if w.ndim == 2:
            return Spectral().init(w)
        if w.ndim in (0, 1):
            return BiasRMS().init(w)


norm_dict = {
    "ColNorm": ColNorm,
    "RowNorm": RowNorm,
    "BiasRMS": BiasRMS,
    "SpectralConv": SpectralConv,
    "Spectral": Spectral,
    "Sign": Sign,
    "Auto": Auto,
}


class Scion(Optimizer):
    r"""Implements Scion, a norm-constrained linear-minimization-oracle optimizer.

    Scion maintains a gradient average and applies the linear minimization
    oracle (LMO) of a chosen norm ball to it, which adapts the update to the
    geometry of the problem. With ``momentum`` denoting one minus the usual
    decay factor, the update is

    .. math::
       \begin{aligned}
       d_t &= (1 - \mu)\, d_{t-1} + \mu\, g_t \\
       \theta_t &= (1 - \gamma)\, \theta_{t-1}
           - \gamma\, \rho\, \mathrm{lmo}_{\mathcal{C}}(d_t)
       \end{aligned}

    where :math:`\gamma` is the learning rate, :math:`\mu` is ``momentum``,
    :math:`\rho` is ``scale`` (the radius of the norm ball :math:`\mathcal{C}`),
    and :math:`\mathrm{lmo}_{\mathcal{C}}` is the linear minimization oracle. The
    weight-shrinkage term :math:`(1 - \gamma)\theta_{t-1}` keeps the iterate
    inside the norm ball; setting ``unconstrained=True`` drops it, giving the
    unconstrained variant

    .. math::
       \theta_t = \theta_{t-1} - \gamma\, \rho\, \mathrm{lmo}_{\mathcal{C}}(d_t).

    The norm ball is selected per parameter group via ``norm`` (one of
    ``'Auto'``, ``'SpectralConv'``, ``'ColNorm'``, ``'RowNorm'``, ``'BiasRMS'``,
    ``'Spectral'``, or ``'Sign'``); for matrix parameters the spectral LMO
    orthogonalizes the gradient average through a Newton-Schulz iteration.

    Reference: Thomas Pethick, Wanyun Xie, Kimon Antonakopoulos, Zhenyu Zhu,
    Antonio Silveti-Falls, Volkan Cevher, "Training Deep Learning Models with
    Norm-Constrained LMOs", ICML 2025.
    https://arxiv.org/abs/2502.07529
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        momentum: float = 1.0,
        norm: str = "Auto",
        norm_kwargs: dict = None,
        scale: float = 1.0,
        unconstrained: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if norm_kwargs is None:
            norm_kwargs = {}
        defaults = dict(
            lr=lr,
            momentum=momentum,
            scale=scale,
            unconstrained=unconstrained,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            scale = group["scale"]
            unconstrained = group["unconstrained"]
            norm_backend = norm_dict[group["norm"]](**group["norm_kwargs"])
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if momentum != 1:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(1 - momentum).add_(g, alpha=momentum)
                    g = buf

                update = scale * norm_backend.lmo(g)
                if not unconstrained:
                    p.data.mul_(1 - lr)
                p.data.add_(update, alpha=-lr)

        return loss

    @torch.no_grad()
    def init(self):
        """Initialize parameters consistently with the chosen norm ball."""
        for group in self.param_groups:
            norm_backend = norm_dict[group["norm"]](**group["norm_kwargs"])
            init_func = norm_backend.init
            scale = group["scale"]
            for p in group["params"]:
                init_func(p)
                p.data *= scale
