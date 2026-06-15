# Adapted from https://github.com/kozistr/pytorch_optimizer (commit 3d08fa0)
# Copyright (c) Hyeongchan Kim (kozistr). Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
r"""Implementation of the Adam-mini optimizer."""

import math

import torch
from torch import distributed as dist

from ...core.optimizer import Optimizer

__all__ = ["AdamMini"]


def _named_parameters(model):
    if isinstance(model, torch.nn.Module):
        return list(model.named_parameters())
    return [(str(i), p) for i, p in enumerate(model)]


class AdamMini(Optimizer):
    r"""Implements Adam-mini, a memory-efficient Adam variant that assigns a
    single second-moment value, and hence a single learning rate, to each
    parameter block.

    .. math::
       \begin{aligned}
            m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t                       \\
            v_{b,t} &= \beta_2 v_{b,t-1} + (1 - \beta_2)
                \mathrm{mean}\!\left(g_{b,t} \odot g_{b,t}\right)      \\
            \hat{m}_t &= m_t / (1 - \beta_1^t), \qquad
                \hat{v}_{b,t} = v_{b,t} / (1 - \beta_2^t)                    \\
            \theta_{b,t} &= \theta_{b,t-1} - \eta\,
                \frac{\hat{m}_{b,t}}{\sqrt{\hat{v}_{b,t}} + \epsilon}
       \end{aligned}

    where the blocks :math:`b` follow the model architecture: embedding and
    output layers keep Adam's coordinate-wise second moment, query and key
    projections use one block per attention head, fused QKV weights use one
    block per head and query group, and every remaining parameter tensor
    forms a single block. Weight decay is decoupled as in AdamW and disabled
    for normalization layers.

    Note:
        The constructor takes the model itself rather than a parameter
        iterable, since the block partition is derived from parameter names.
        A plain iterable of tensors is also accepted; its entries are treated
        as unnamed, one block per tensor.

    Reference: Yushun Zhang, Congliang Chen, Ziniu Li, Tian Ding, Chenwei Wu,
    Diederik P. Kingma, Yinyu Ye, Zhi-Quan Luo, Ruoyu Sun,
    "Adam-mini: Use Fewer Learning Rates To Gain More", ICLR 2025.
    https://arxiv.org/abs/2406.16793
    """

    def __init__(
        self,
        model,
        lr: float = 1.0,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.1,
        model_sharding: bool = False,
        num_embeds: int = 2048,
        num_heads: int = 32,
        num_query_groups: int | None = None,
        eps: float = 1e-8,
        maximize: bool = False,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        for i, beta in enumerate(betas):
            if not 0.0 <= beta < 1.0:
                raise ValueError(f"Invalid beta parameter at index {i}: {beta}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if num_embeds < 0:
            raise ValueError(f"Invalid num_embeds value: {num_embeds}")
        if num_heads < 0:
            raise ValueError(f"Invalid num_heads value: {num_heads}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        self.num_query_groups = (
            num_query_groups if num_query_groups is not None else num_embeds
        )
        if num_embeds % self.num_query_groups != 0:
            raise ValueError(
                f"num_embeds ({num_embeds}) must be divisible by "
                f"num_query_groups ({self.num_query_groups})"
            )

        self.world_size = torch.cuda.device_count()

        self.model_sharding = model_sharding
        self.num_embeds = num_embeds
        self.num_heads = num_heads
        self.maximize = maximize

        self.embed_blocks = {"embed", "embd", "wte", "lm_head.weight", "output.weight"}
        self.qk_blocks = {"k_proj.weight", "q_proj.weight", "wq.weight", "wk.weight"}

        self.named_params = _named_parameters(model)

        groups = self.get_optimizer_groups(weight_decay)

        defaults = {"lr": lr, "betas": betas, "eps": eps}
        super().__init__(groups, defaults)

    def get_optimizer_groups(self, weight_decay: float):
        groups = []
        for name, param in self.named_params:
            if not param.requires_grad:
                continue

            group = {
                "name": name,
                "params": param,
                "weight_decay": 0.0 if ("norm" in name or "ln_f" in name) else weight_decay,
            }

            if any(block in name for block in self.qk_blocks):
                group["parameter_per_head"] = self.num_embeds * self.num_embeds // self.num_heads

            if "attn.attn.weight" in name or "attn.qkv.weight" in name:
                group["n_head"] = self.num_heads
                group["q_per_kv"] = self.num_embeds // self.num_query_groups

            groups.append(group)

        return groups

    @staticmethod
    def step_embed(
        p,
        grad,
        state,
        lr: float,
        beta1: float,
        beta2: float,
        bias_correction1: float,
        bias_correction2_sq: float,
        eps: float,
    ) -> None:
        if len(state) == 0:
            state["m"] = torch.zeros_like(p, dtype=torch.float32)
            state["v"] = torch.zeros_like(p, dtype=torch.float32)

        m, v = state["m"], state["v"]

        m.lerp_(grad, weight=1.0 - beta1)
        v.mul_(beta2).addcmul_(grad, grad.conj(), value=1.0 - beta2)

        h = (v.sqrt() / bias_correction2_sq).add_(eps)

        p.addcdiv_(m, h, value=-lr / bias_correction1)

    @staticmethod
    def step_attn_proj(
        p,
        grad,
        state,
        parameter_per_head: int,
        lr: float,
        beta1: float,
        beta2: float,
        bias_correction1: float,
        bias_correction2_sq: float,
        eps: float,
    ) -> None:
        if len(state) == 0:
            state["m"] = torch.zeros_like(p, dtype=torch.float32).view(-1, parameter_per_head)
            state["head"] = state["m"].shape[0]
            state["v_mean"] = torch.zeros(state["head"], device=state["m"].device)

        m, v = state["m"], state["v_mean"]

        head = state["head"]
        grad = grad.view(head, parameter_per_head)

        m.lerp_(grad, weight=1.0 - beta1)

        tmp_lr = torch.mean(grad * grad, dim=1).to(m.device)
        v.mul_(beta2).add_(tmp_lr, alpha=1.0 - beta2)

        h = (v.sqrt() / bias_correction2_sq).add_(eps)

        update = (1 / (h * bias_correction1)).view(head, 1).mul_(m)

        if p.dim() > 1:
            d0, d1 = p.size()
            update = update.view(d0, d1)
        else:
            update = update.view(-1)

        p.add_(update, alpha=-lr)

    @staticmethod
    def step_attn(
        p,
        grad,
        state,
        num_heads: int,
        q_per_kv: int,
        lr: float,
        beta1: float,
        beta2: float,
        bias_correction1: float,
        bias_correction2_sq: float,
        eps: float,
    ) -> None:
        if len(state) == 0:
            state["m"] = torch.zeros_like(p, dtype=torch.float32).view(num_heads, q_per_kv + 2, -1)
            state["v_mean"] = torch.zeros(num_heads, q_per_kv + 2, device=state["m"].device)

        m, v = state["m"], state["v_mean"]

        grad = grad.view(num_heads, q_per_kv + 2, -1)

        m.lerp_(grad, weight=1.0 - beta1)

        tmp_lr = torch.mean(grad * grad, dim=2).to(m.device)
        v.mul_(beta2).add_(tmp_lr, alpha=1.0 - beta2)

        h = (v.sqrt() / bias_correction2_sq).add_(eps)

        update = (1 / (h * bias_correction1)).view(num_heads, q_per_kv + 2, -1).mul_(m)

        if p.dim() > 1:
            d0, d1 = p.size()
            update = update.view(d0, d1)
        else:
            update = update.view(-1)

        p.add_(update, alpha=-lr)

    def step_lefts(
        self,
        p,
        grad,
        state,
        lr: float,
        beta1: float,
        beta2: float,
        bias_correction1: float,
        bias_correction2_sq: float,
        eps: float,
    ) -> None:
        if len(state) == 0:
            dim = torch.tensor(p.numel(), device=p.device, dtype=torch.float32)

            reduced = False
            if self.model_sharding and self.world_size > 1:
                tensor_list = [torch.zeros_like(dim) for _ in range(self.world_size)]
                dist.all_gather(tensor_list, dim)

                s, dim = 0, 0
                for d in tensor_list:
                    if d > 0:
                        s += 1
                    dim += d

                if s >= 2:
                    reduced = True

            state["m"] = torch.zeros_like(p, dtype=torch.float32)
            state["v_mean"] = torch.tensor(0.0, device=state["m"].device)
            state["dimension"] = dim
            state["reduced"] = reduced

        tmp_lr = torch.sum(grad * grad)

        if state["reduced"]:
            dist.all_reduce(tmp_lr, op=dist.ReduceOp.SUM)

        tmp_lr.div_(state["dimension"])

        m, v = state["m"], state["v_mean"]

        m.lerp_(grad, weight=1.0 - beta1)
        v.mul_(beta2).add_(tmp_lr, alpha=1.0 - beta2)

        h = (v.sqrt() / bias_correction2_sq).add_(eps)

        stepsize = (1 / bias_correction1) / h

        update = m * stepsize

        p.add_(update, alpha=-lr)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if "step" not in group:
                group["step"] = 0
            group["step"] += 1

            name = group["name"]

            beta1, beta2 = group["betas"]

            bias_correction1 = 1.0 - beta1 ** group["step"]
            bias_correction2_sq = math.sqrt(1.0 - beta2 ** group["step"])

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamMini does not support sparse gradients")
                if torch.is_complex(p):
                    raise RuntimeError("AdamMini does not support complex parameters")

                grad = grad.to(torch.float32)

                if self.maximize:
                    grad = -grad

                state = self.state[p]

                p.mul_(1.0 - group["weight_decay"] * group["lr"])

                if any(block in name for block in self.embed_blocks):
                    self.step_embed(
                        p,
                        grad,
                        state,
                        group["lr"],
                        beta1,
                        beta2,
                        bias_correction1,
                        bias_correction2_sq,
                        group["eps"],
                    )
                elif any(block in name for block in self.qk_blocks):
                    self.step_attn_proj(
                        p,
                        grad,
                        state,
                        group["parameter_per_head"],
                        group["lr"],
                        beta1,
                        beta2,
                        bias_correction1,
                        bias_correction2_sq,
                        group["eps"],
                    )
                elif "attn.attn.weight" in name or "attn.qkv.weight" in name:
                    self.step_attn(
                        p,
                        grad,
                        state,
                        group["n_head"],
                        group["q_per_kv"],
                        group["lr"],
                        beta1,
                        beta2,
                        bias_correction1,
                        bias_correction2_sq,
                        group["eps"],
                    )
                else:
                    self.step_lefts(
                        p,
                        grad,
                        state,
                        group["lr"],
                        beta1,
                        beta2,
                        bias_correction1,
                        bias_correction2_sq,
                        group["eps"],
                    )

        return loss
