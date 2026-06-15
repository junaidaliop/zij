# Adapted from https://github.com/zichongli5/NorMuon (commit c6989a8)
# Copyright (c) 2025 zichongli5. Licensed under the MIT license.
# See THIRD_PARTY_NOTICES.md for details.
"""Implementation of the NorMuon optimizer."""

import torch
import torch.distributed as dist

from ...core.optimizer import Optimizer

__all__ = [
    "NorMuon",
    "SingleDeviceNorMuon",
    "NorMuonWithAuxAdam",
    "SingleDeviceNorMuonWithAuxAdam",
]


def zeropower_via_newtonschulz5(g, steps=5):
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of
    G.

    A quintic iteration whose coefficients are selected to maximize the slope at
    zero. For the purpose of minimizing steps, it is empirically effective to keep
    increasing the slope at zero even beyond the point where the iteration no
    longer converges all the way to one everywhere on the interval. This iteration
    therefore does not produce :math:`UV^\\top` but rather something like
    :math:`US'V^\\top` where :math:`S'` is diagonal with entries roughly uniform on
    ``[0.5, 1.5]``, which turns out not to hurt model performance at all relative
    to :math:`UV^\\top`, where :math:`USV^\\top = G` is the SVD.
    """
    assert g.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.bfloat16()
    if g.size(-2) > g.size(-1):
        x = x.mT

    # Ensure spectral norm is at most 1.
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations.
    for _ in range(steps):
        s = x @ x.mT
        t = b * s + c * s @ s
        x = a * x + t @ x

    if g.size(-2) > g.size(-1):
        x = x.mT
    return x


def normuon_update(
    grad, momentum, second_momentum, beta=0.95, beta2=0.95, ns_steps=5, nesterov=True
):
    """Compute one NorMuon update from the gradient and the running buffers."""
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    original_shape = None
    if update.ndim == 4:  # convolutional filters
        original_shape = update.shape
        update = update.reshape(update.size(0), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update = update.to(grad.dtype)

    if original_shape is not None:
        update = update.reshape(original_shape)

    # Neuron-wise (row-wise) second-moment normalization, rescaled so the update
    # Frobenius norm matches the pre-normalization orthogonalized update.
    vnorm = update.norm(dim=(-2, -1), keepdim=True)
    v_mean = torch.mean(update * update, dim=-1, keepdim=True)
    second_momentum.lerp_(v_mean, 1 - beta2)
    step_size = 1 / second_momentum.sqrt().add_(1e-10)
    update.mul_(step_size)
    vnorm_new = update.norm(dim=(-2, -1), keepdim=True)
    update.mul_(vnorm / (vnorm_new.add_(1e-10)))

    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class NorMuon(Optimizer):
    r"""Implements NorMuon, neuron-wise normalized Muon.

    NorMuon augments Muon with a per-row (neuron-wise) second moment applied to
    the orthogonalized update, then rescales so the overall update norm matches
    Muon's. For a matrix parameter, the momentum :math:`M_t` is orthogonalized
    through a Newton-Schulz iteration, a per-row second moment :math:`v_t` is
    accumulated on the squared rows of the orthogonalized update, and each row is
    divided by its root-second-moment before a Frobenius-norm-preserving rescale:

    .. math::
       \begin{aligned}
       M_t &= \beta_1 M_{t-1} + (1 - \beta_1) G_t \\
       O_t &= \mathrm{NewtonSchulz}\big((1 - \beta_1) G_t + \beta_1 M_t\big) \\
       v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\,
           \tfrac{1}{n} \sum_{j} (O_t)_{:,j}^2 \\
       \tilde{O}_t &= O_t \oslash \big(\sqrt{v_t} + \epsilon\big) \\
       \hat{O}_t &= \tilde{O}_t \,
           \frac{\lVert O_t \rVert_F}{\lVert \tilde{O}_t \rVert_F}
           \sqrt{\max\!\left(1, \tfrac{m}{n}\right)} \\
       \theta_t &= \theta_{t-1} - \gamma\, \hat{O}_t
       \end{aligned}

    where :math:`m, n` are the matrix dimensions, :math:`v_t` and the division
    :math:`\oslash` are taken per row (one value per output neuron), and the
    Nesterov-style momentum is enabled by default. This distributed variant shards
    parameters across ranks of the default process group and all-gathers the
    updated weights, so it requires an initialized :mod:`torch.distributed` group.

    Reference: Zichong Li, Liming Liu, Chen Liang, Weizhu Chen, Tuo Zhao,
    "NorMuon: Making Muon more efficient and scalable", 2025.
    https://arxiv.org/abs/2510.05491
    """

    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, beta2=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2)
        assert (
            isinstance(params, list)
            and len(params) >= 1
            and isinstance(params[0], torch.nn.Parameter)
        )
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (
                dist.get_world_size() - len(params) % dist.get_world_size()
            )
            for base_i in range(len(params))[:: dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    had_grad = p.grad is not None
                    if not had_grad:
                        p.grad = torch.zeros_like(p)  # force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])
                    update = normuon_update(
                        p.grad,
                        state["momentum_buffer"],
                        state["second_momentum_buffer"],
                        beta=group["momentum"],
                        beta2=group["beta2"],
                    )
                    if group["weight_decay"] and had_grad:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(
                    params_pad[base_i : base_i + dist.get_world_size()],
                    params_pad[base_i + dist.get_rank()],
                )

        return loss


class SingleDeviceNorMuon(Optimizer):
    r"""Single-device NorMuon for use in non-distributed settings.

    Same update rule as :class:`NorMuon` (see its docstring for the math) without
    the parameter sharding and all-gather, so it runs on a single process.

    Reference: Zichong Li, Liming Liu, Chen Liang, Weizhu Chen, Tuo Zhao,
    "NorMuon: Making Muon more efficient and scalable", 2025.
    https://arxiv.org/abs/2510.05491
    """

    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, beta2=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                had_grad = p.grad is not None
                if not had_grad:
                    p.grad = torch.zeros_like(p)  # force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])
                update = normuon_update(
                    p.grad,
                    state["momentum_buffer"],
                    state["second_momentum_buffer"],
                    beta=group["momentum"],
                    beta2=group["beta2"],
                )
                if group["weight_decay"] and had_grad:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


def adam_update(grad, buf1, buf2, step, betas, eps):
    """Standard Adam update used for the auxiliary parameter groups."""
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class NorMuonWithAuxAdam(Optimizer):
    r"""Distributed NorMuon paired with an auxiliary Adam optimizer.

    Parameter groups with ``use_muon=True`` are updated with NorMuon (see
    :class:`NorMuon` for the math); groups with ``use_muon=False`` are updated with
    decoupled-weight-decay Adam, so embeddings, the classifier head, and gains or
    biases can share the optimizer. This variant shards the NorMuon groups across
    ranks of the default process group, so it requires an initialized
    :mod:`torch.distributed` group.

    Reference: Zichong Li, Liming Liu, Chen Liang, Weizhu Chen, Tuo Zhao,
    "NorMuon: Making Muon more efficient and scalable", 2025.
    https://arxiv.org/abs/2510.05491
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(
                    group["params"], key=lambda x: x.size(), reverse=True
                )
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["beta2"] = group.get("beta2", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {
                    "params",
                    "lr",
                    "momentum",
                    "beta2",
                    "weight_decay",
                    "use_muon",
                }
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {
                    "params",
                    "lr",
                    "betas",
                    "eps",
                    "weight_decay",
                    "use_muon",
                }
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (
                    dist.get_world_size() - len(params) % dist.get_world_size()
                )
                for base_i in range(len(params))[:: dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        had_grad = p.grad is not None
                        if not had_grad:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                            state["second_momentum_buffer"] = torch.zeros_like(
                                p[..., 0:1]
                            )
                        update = normuon_update(
                            p.grad,
                            state["momentum_buffer"],
                            state["second_momentum_buffer"],
                            beta=group["momentum"],
                            beta2=group["beta2"],
                        )
                        if group["weight_decay"] and had_grad:
                            p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(
                        params_pad[base_i : base_i + dist.get_world_size()],
                        params_pad[base_i + dist.get_rank()],
                    )
            else:
                for p in group["params"]:
                    had_grad = p.grad is not None
                    if not had_grad:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    if group["weight_decay"] and had_grad:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceNorMuonWithAuxAdam(Optimizer):
    r"""Single-device counterpart to :class:`NorMuonWithAuxAdam`.

    Groups with ``use_muon=True`` are updated with NorMuon (see :class:`NorMuon`
    for the math); groups with ``use_muon=False`` are updated with
    decoupled-weight-decay Adam. Runs on a single process without sharding.

    Reference: Zichong Li, Liming Liu, Chen Liang, Weizhu Chen, Tuo Zhao,
    "NorMuon: Making Muon more efficient and scalable", 2025.
    https://arxiv.org/abs/2510.05491
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["beta2"] = group.get("beta2", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {
                    "params",
                    "lr",
                    "momentum",
                    "beta2",
                    "weight_decay",
                    "use_muon",
                }
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {
                    "params",
                    "lr",
                    "betas",
                    "eps",
                    "weight_decay",
                    "use_muon",
                }
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    had_grad = p.grad is not None
                    if not had_grad:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])
                    update = normuon_update(
                        p.grad,
                        state["momentum_buffer"],
                        state["second_momentum_buffer"],
                        beta=group["momentum"],
                        beta2=group["beta2"],
                    )
                    if group["weight_decay"] and had_grad:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    had_grad = p.grad is not None
                    if not had_grad:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    if group["weight_decay"] and had_grad:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
