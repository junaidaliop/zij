# Adapted from https://github.com/timlautk/polargrad (commit 1f1e5be)
# Copyright (c) 2025 Tim Tsz-Kit Lau. Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the PolarGrad optimizer."""

from itertools import repeat
from typing import Optional, Tuple

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["PolarGrad"]


def zeropower_via_newtonschulz5(
    g: torch.Tensor,
    compute_hermitian: bool = False,
    max_iterations: int = 5,
    a: float = 3.4445,
    b: float = -4.7750,
    c: float = 2.0315,
):
    r"""Compute the orthogonal polar factor of ``g`` via the quintic
    Newton-Schulz iteration.

    The coefficients are selected to maximize the slope at zero. The iteration
    does not produce :math:`UV^\top` exactly but rather :math:`US'V^\top` with
    :math:`S'` diagonal whose entries are roughly uniform in ``[0.5, 1.5]``,
    which does not hurt model performance relative to :math:`UV^\top` from the
    singular value decomposition :math:`USV^\top = g`.
    """
    if g.ndim < 2:
        raise ValueError(f"input must be at least 2-dimensional. got {g.ndim}D.")
    x = g.bfloat16()
    transpose = g.size(-2) > g.size(-1)
    if transpose:
        x = x.mT

    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(max_iterations):
        s = x @ x.mT
        t = b * s + c * s @ s
        x = a * x + t @ x

    h = None
    if compute_hermitian:
        h = g.type_as(x).mT @ x.mT
        h = (h + h.mT) / 2

    if transpose:
        x = x.mT
        if compute_hermitian:
            h = h.mT

    if compute_hermitian:
        return x, h
    return x


_POLAR_EXPRESS_COEFFS = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]
# Safety factor for numerical stability, applied to all but the last polynomial.
_POLAR_EXPRESS_COEFFS = [
    (a / 1.01, b / 1.01 ** 3, c / 1.01 ** 5)
    for (a, b, c) in _POLAR_EXPRESS_COEFFS[:-1]
] + [_POLAR_EXPRESS_COEFFS[-1]]


def polar_express(
    g: torch.Tensor,
    compute_hermitian: bool = False,
    max_iterations: int = 5,
):
    r"""Compute the orthogonal polar factor of ``g`` via the Polar Express
    iteration, a step-dependent quintic schedule that approximates the optimal
    matrix sign method.

    Reference: Noah Amsel, David Persson, Christopher Musco, Robert M. Gower,
    "The Polar Express: Optimal Matrix Sign Methods and Their Application to the
    Muon Algorithm", 2025. https://arxiv.org/abs/2505.16932
    """
    if g.ndim < 2:
        raise ValueError(f"input must be at least 2-dimensional. got {g.ndim}D.")
    x = g.bfloat16()
    transpose = g.size(-2) > g.size(-1)
    if transpose:
        x = x.mT

    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    schedule = _POLAR_EXPRESS_COEFFS[:max_iterations] + list(
        repeat(_POLAR_EXPRESS_COEFFS[-1], max_iterations - len(_POLAR_EXPRESS_COEFFS))
    )
    for a, b, c in schedule:
        s = x @ x.mT
        t = b * s + c * s @ s
        x = a * x + t @ x

    h = None
    if compute_hermitian:
        h = g.type_as(x).mT @ x.mT
        h = (h + h.mT) / 2

    if transpose:
        x = x.mT
        if compute_hermitian:
            h = h.mT

    if compute_hermitian:
        return x, h
    return x


def _use_qr(u, params):
    a_minus_e_by_sqrt_c, sqrt_c, e = params
    n = u.size(1)
    eye = torch.eye(n, dtype=u.dtype, device=u.device)
    y = torch.cat([sqrt_c * u, eye], dim=0)
    q, _ = torch.linalg.qr(y, mode="reduced")
    q1 = q[: u.size(0), :n]
    q2 = q[u.size(0):, :].mT.conj()
    return e * u + a_minus_e_by_sqrt_c * (q1 @ q2)


def _use_cholesky(u, params):
    a_minus_e, c, e = params
    n = u.size(1)
    eye = torch.eye(n, dtype=u.dtype, device=u.device)
    x = c * (u.mT.conj() @ u) + eye
    y, _ = torch.linalg.cholesky_ex(x)
    z = torch.linalg.solve_triangular(
        y.conj(), u.mT, upper=False, left=True, unitriangular=False
    ).conj()
    z = torch.linalg.solve_triangular(
        y.conj().mT, z, upper=False, left=True, unitriangular=False
    ).mT.conj()
    return e * u + a_minus_e * z


def _qdwh(x, max_iterations, eps):
    if eps is None:
        eps = float(torch.finfo(x.dtype).eps)

    one_norm = torch.linalg.norm(x, ord=1)
    inf_norm = torch.linalg.norm(x, ord=float("inf"))
    alpha_inverse = 1 / (one_norm * inf_norm) ** 0.5
    alpha_inverse = torch.where(one_norm == 0, torch.ones_like(alpha_inverse), alpha_inverse)
    u = x * alpha_inverse

    l = eps
    tol_l = 10.0 * eps / 2.0
    tol_norm = tol_l ** (1 / 3)

    def get_qr_params(a, b, c):
        e = b / c
        sqrt_c = c ** 0.5
        return ((a - e) / sqrt_c, sqrt_c, e)

    def get_chol_params(a, b, c):
        e = b / c
        return (a - e, c, e)

    cholesky_cutoff = 100
    qr_coefs = []
    chol_coefs = []
    k = 0
    while l + tol_l < 1 and k < max_iterations:
        k += 1
        l2 = l * l
        dd = (4 * (1 / l2 - 1) / l2) ** (1 / 3)
        sqd = (1.0 + dd) ** 0.5
        a = sqd + (2 - dd + 2 * (2 - l2) / (l2 * sqd)) ** 0.5
        b = (a - 1) ** 2 / 4
        c = a + b - 1
        l = l * (a + b * l2) / (1 + c * l2)
        if c > cholesky_cutoff:
            qr_coefs.append(get_qr_params(a, b, c))
        else:
            chol_coefs.append(get_chol_params(a, b, c))

    for params in qr_coefs:
        u = _use_qr(u, params)

    is_not_converged = True
    for params in chol_coefs:
        u_prev = u.clone()
        u = _use_cholesky(u, params)
        is_not_converged = torch.linalg.matrix_norm(u - u_prev) > tol_norm

    k_counter = len(qr_coefs) + len(chol_coefs)
    halley = get_chol_params(3, 1, 3)
    while is_not_converged and k_counter < max_iterations:
        u_prev = u.clone()
        u = _use_cholesky(u, halley)
        is_not_converged = torch.linalg.matrix_norm(u - u_prev) > tol_norm
        k_counter += 1

    u = 1.5 * u - 0.5 * u @ (u.mT.conj() @ u)
    return u


def qdwh(x, compute_hermitian=False, max_iterations=None, eps=None):
    r"""QR-based dynamically weighted Halley iteration for polar decomposition.

    Modified from the JAX implementation
    (https://github.com/jax-ml/jax/blob/main/jax/_src/lax/qdwh.py).

    Reference: Yuji Nakatsukasa, Zhaojun Bai, François Gygi, "Optimizing
    Halley's iteration for computing the matrix polar decomposition", SIAM
    Journal on Matrix Analysis and Applications 31(5):2700-2720, 2010.
    """
    if max_iterations is None:
        max_iterations = 10
    u = _qdwh(x, max_iterations, eps)
    if compute_hermitian:
        h = u.mT.conj() @ x
        h = (h + h.mT.conj()) / 2
        return u, h
    return u


def polar(
    a: torch.Tensor,
    *,
    method: str = "qdwh",
    compute_hermitian: bool = False,
    eps: Optional[float] = None,
    max_iterations: Optional[int] = None,
    ns_coeffs: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
):
    r"""Compute the polar decomposition of a matrix.

    Given an :math:`m \times n` matrix ``a``, returns the orthogonal polar
    factor ``u`` and (optionally) the symmetric positive-semidefinite factor
    ``h`` such that :math:`a = u h` (when :math:`m \ge n`) or :math:`a = h u`
    (when :math:`m < n`). Three backends are supported via ``method``:
    ``'ns'`` (Newton-Schulz), ``'qdwh'`` (QR-based dynamically weighted
    Halley), and ``'polar_express'`` (the Polar Express schedule).
    """
    arr = torch.as_tensor(a)
    if arr.ndim != 2:
        raise ValueError("the input `a` must be a 2-D array.")

    m, n = arr.shape
    max_iterations = max_iterations if max_iterations is not None else 5

    if method == "qdwh":
        if m >= n:
            res = qdwh(
                arr,
                compute_hermitian=compute_hermitian,
                max_iterations=max_iterations,
                eps=eps,
            )
            if compute_hermitian:
                unitary, posdef = res
            else:
                unitary = res
        else:
            arr_t = arr.mT.conj()
            res = qdwh(
                arr_t,
                compute_hermitian=compute_hermitian,
                max_iterations=max_iterations,
                eps=eps,
            )
            if compute_hermitian:
                unitary, posdef = res
                unitary = unitary.mT.conj()
                posdef = posdef.mT.conj()
            else:
                unitary = res.mT.conj()
    elif method == "ns":
        res = zeropower_via_newtonschulz5(
            arr,
            compute_hermitian=compute_hermitian,
            max_iterations=max_iterations,
            a=ns_coeffs[0],
            b=ns_coeffs[1],
            c=ns_coeffs[2],
        )
        if compute_hermitian:
            unitary, posdef = res
        else:
            unitary = res
    elif method == "polar_express":
        res = polar_express(
            arr, compute_hermitian=compute_hermitian, max_iterations=max_iterations
        )
        if compute_hermitian:
            unitary, posdef = res
        else:
            unitary = res
    else:
        raise ValueError(f"unknown polar decomposition method {method!r}.")

    return (unitary, posdef) if compute_hermitian else (unitary,)


class PolarGrad(Optimizer):
    r"""Implements PolarGrad, a polar-decomposition preconditioned optimizer.

    For a matrix parameter, PolarGrad orthogonalizes the gradient (or its
    momentum average) through the polar decomposition and rescales the
    orthogonal factor by the nuclear norm of the same matrix. Writing
    :math:`U_t H_t = \mathrm{polar}(M_t)` for the polar decomposition, the
    nuclear norm equals :math:`\mathrm{tr}(H_t) = \langle M_t, U_t
    \rangle_F`, which avoids a full singular value decomposition. With
    exponential-moving-average momentum and decoupled weight decay, the
    momentum-first update (the default) is

    .. math::
       \begin{aligned}
       M_t &= \beta M_{t-1} + (1 - \beta) G_t \\
       U_t H_t &= \mathrm{polar}(M_t) \\
       \theta_t &= (1 - \lambda \gamma)\, \theta_{t-1}
           - \gamma\, \mathrm{tr}(H_t)\, U_t
       \end{aligned}

    where :math:`\gamma` is the learning rate, :math:`\beta` is ``momentum``,
    and :math:`\lambda` is ``weight_decay``. Setting ``polar_first=True``
    selects the polar-first variant, which decomposes the gradient before the
    momentum average,

    .. math::
       U_t H_t = \mathrm{polar}(G_t), \quad
       M_t = \beta M_{t-1} + (1 - \beta) U_t, \quad
       \theta_t = (1 - \lambda \gamma)\, \theta_{t-1}
           - \gamma\, \mathrm{tr}(H_t)\, M_t.

    The nuclear-norm scaling subsumes Muon, whose orthogonalized update PolarGrad
    recovers when the scaling is dropped. The orthogonal polar factor is computed
    by the backend named in ``method`` (``'qdwh'``, ``'ns'``, or
    ``'polar_express'``).

    Note: only matrix (two-dimensional) parameters are supported; pair PolarGrad
    with another optimizer for embeddings, biases, and scalar parameters.

    Reference: Tim Tsz-Kit Lau, Qi Long, Weijie Su, "PolarGrad: A Class of
    Matrix-Gradient Optimizers from a Unifying Preconditioning Perspective",
    2025. https://arxiv.org/abs/2505.21799
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 2e-2,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        polar_first: bool = False,
        method: str = "qdwh",
        inner_steps: int = 2,
        a: float = 3.4445,
        b: float = -4.7750,
        c: float = 2.0315,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if inner_steps <= 0:
            raise ValueError(f"Invalid inner_steps value: {inner_steps}")
        if method not in ("qdwh", "ns", "polar_express"):
            raise ValueError(f"Invalid method: {method!r}")
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            polar_first=polar_first,
            method=method,
            inner_steps=inner_steps,
            a=a,
            b=b,
            c=c,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("PolarGrad does not support sparse gradients")
                if g.ndim != 2:
                    raise ValueError(
                        "PolarGrad only supports 2-dimensional parameters; "
                        f"got a {g.ndim}D parameter"
                    )
                state = self.state[p]
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(g)
                m = state["momentum"]

                ns_coeffs = (group["a"], group["b"], group["c"])
                if group["polar_first"]:
                    u = polar(
                        g,
                        method=group["method"],
                        max_iterations=group["inner_steps"],
                        ns_coeffs=ns_coeffs,
                    )[0]
                    nuc_norm = torch.sum(g.type_as(u) * u)
                    m.lerp_(u, 1 - group["momentum"])
                    update = nuc_norm * m
                else:
                    m.lerp_(g, 1 - group["momentum"])
                    u = polar(
                        m,
                        method=group["method"],
                        max_iterations=group["inner_steps"],
                        ns_coeffs=ns_coeffs,
                    )[0]
                    nuc_norm = torch.sum(m.type_as(u) * u)
                    update = nuc_norm * u

                p.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    update.type_as(p), alpha=-group["lr"]
                )

        return loss
