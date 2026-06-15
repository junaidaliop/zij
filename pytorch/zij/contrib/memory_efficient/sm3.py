# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa0)
# Copyright (c) Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# A PyTorch port of the official TensorFlow SM3 from google-research/google-research.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the SM3 optimizer."""

import torch

from ...core.optimizer import Optimizer, ParamsT

__all__ = ["SM3"]


@torch.no_grad()
def _reduce_max_except_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Reduce-max over all dimensions of ``x`` except ``dim``."""
    rank = len(x.shape)
    if rank == 0:
        return x
    if dim >= rank:
        raise ValueError(f"Given dim is bigger than rank: {dim} >= {rank}")

    for d in range(rank):
        if d != dim:
            x = x.max(dim=d, keepdim=True).values
    return x


class SM3(Optimizer):
    r"""Implements SM3, the memory-efficient adaptive method of Anil et al.

    This is the SM3-II variant. For a parameter tensor, the cover sets
    :math:`S_r` are its slices along each axis, so a :math:`d_1 \times d_2`
    matrix keeps :math:`d_1 + d_2` accumulator entries instead of
    :math:`d_1 d_2`:

    .. math::
       \begin{aligned}
            \nu_t(j) &= \min_{r : S_r \ni j} \mu_{t-1}(r) + g_t(j)^2 \\
            \theta_{t+1}(j) &= \theta_t(j) - \eta \,
                \frac{g_t(j)}{\sqrt{\nu_t(j)}} \\
            \mu_t(r) &= \max_{j \in S_r} \nu_t(j)
       \end{aligned}

    Note:
        The defaults follow the paper: with ``beta=0`` the accumulators upper
        bound the running sums of squared gradients. Setting ``beta > 0``
        replaces the sums with exponential moving averages, and
        ``momentum > 0`` adds a moving average of the preconditioned update,
        at the cost of one extra buffer per parameter. Momentum is ignored
        for sparse gradients.

    Reference: Rohan Anil, Vineet Gupta, Tomer Koren, Yoram Singer,
    "Memory-Efficient Adaptive Optimization", NeurIPS 2019.
    https://arxiv.org/abs/1901.11150
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-1,
        momentum: float = 0.0,
        beta: float = 0.0,
        eps: float = 1e-30,
        maximize: bool = False,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"Invalid beta value: {beta}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "beta": beta,
            "eps": eps,
            "maximize": maximize,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)

    @staticmethod
    def _make_sparse(grad: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        if grad._indices().dim() == 0 or values.dim() == 0:
            return grad.new().resize_as_(grad)
        return grad.new(grad._indices(), values, grad.size())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum, beta = group["momentum"], group["beta"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if torch.is_complex(p):
                    raise RuntimeError("SM3 does not support complex parameters")
                if group["maximize"]:
                    grad = -grad

                shape = grad.shape
                rank = len(shape)

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                    if grad.is_sparse:
                        state["accumulator_0"] = torch.zeros(
                            shape[0], dtype=grad.dtype, device=grad.device
                        )
                    elif rank == 0:
                        state["accumulator_0"] = torch.zeros_like(grad)
                    else:
                        for i in range(rank):
                            state[f"accumulator_{i}"] = torch.zeros(
                                [1] * i + [shape[i]] + [1] * (rank - 1 - i),
                                dtype=grad.dtype,
                                device=grad.device,
                            )

                if grad.is_sparse:
                    grad = grad.coalesce()

                    acc = state["accumulator_0"]
                    update_values = torch.gather(acc, 0, grad._indices()[0])
                    if beta > 0.0:
                        update_values.mul_(beta)
                    update_values.addcmul_(grad._values(), grad._values(), value=1.0 - beta)

                    nu_max = _reduce_max_except_dim(
                        self._make_sparse(grad, update_values).to_dense(), 0
                    ).squeeze_()

                    if beta > 0.0:
                        torch.max(acc, nu_max, out=acc)
                    else:
                        acc.copy_(nu_max)

                    update_values.add_(group["eps"]).rsqrt_().mul_(grad._values())

                    update = self._make_sparse(grad, update_values)
                else:
                    update = state["accumulator_0"].clone()
                    for i in range(1, rank):
                        update = torch.min(update, state[f"accumulator_{i}"])

                    if beta > 0.0:
                        update.mul_(beta)
                    update.addcmul_(grad, grad, value=1.0 - beta)

                    for i in range(rank):
                        acc = state[f"accumulator_{i}"]
                        nu_max = _reduce_max_except_dim(update, i)
                        if beta > 0.0:
                            torch.max(acc, nu_max, out=acc)
                        else:
                            acc.copy_(nu_max)

                    update.add_(group["eps"]).rsqrt_().mul_(grad)

                    if momentum > 0.0:
                        m = state["momentum_buffer"]
                        m.mul_(momentum).add_(update, alpha=1.0 - momentum)
                        update = m

                p.add_(update, alpha=-group["lr"])

        return loss
