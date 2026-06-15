# Adapted from https://github.com/OpenLMLab/LOMO (commit 45d4bac)
# Copyright (c) 2023 OpenLMLab. Licensed under the MIT license.
# See THIRD_PARTY_NOTICES.md for details.
"""Implementations of the LOMO and AdaLomo optimizers."""

import math
import warnings

import torch
import torch.distributed as dist

from ...core.optimizer import Optimizer

__all__ = ["Lomo", "AdaLomo"]


def _named_parameters(model):
    if isinstance(model, torch.nn.Module):
        return list(model.named_parameters())
    return [(str(i), p) for i, p in enumerate(model)]


class Lomo(Optimizer):
    r"""Implements LOMO, low-memory optimization fusing the SGD update into backward.

    .. math::
       \theta_{t+1} = \theta_t - \eta\, g_t

    The update is applied to each parameter in place as soon as its gradient
    is computed during the backward pass, and the gradient is freed
    immediately afterward, so the full gradient is never materialized.
    Nonzero ``weight_decay`` :math:`\lambda` is decoupled, multiplying the
    parameter by :math:`1 - \eta\lambda` before the gradient step. Gradient
    norm clipping needs the global norm before any parameter is touched, so
    it takes two backward passes: one to gather the norm and one to update.

    Reference: Kai Lv, Yuqing Yang, Tengxiao Liu, Qinghui Gao, Qipeng Guo,
    Xipeng Qiu, "Full Parameter Fine-tuning for Large Language Models with
    Limited Resources", ACL 2024.
    https://arxiv.org/abs/2306.09782

    Note:
        Drive training with :meth:`fused_backward` instead of
        ``loss.backward()`` followed by ``step()``. When ``clip_grad_norm``
        is set, call :meth:`grad_norm` on the loss first. fp16 losses enable
        a dynamic loss scaler automatically.
    """

    def __init__(
        self,
        model,
        lr=1e-3,
        clip_grad_norm=None,
        clip_grad_value=None,
        weight_decay=0.0,
        loss_scale_args={},
    ):
        self.named_params = _named_parameters(model)
        self.lr = lr
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value
        self.loss_scaler = None
        self.loss_scale_args = loss_scale_args
        self.weight_decay = weight_decay
        if self.weight_decay > 0.0:
            self.do_weight_decay = True
        else:
            self.do_weight_decay = False

        # for grad norm
        if self.clip_grad_norm is not None and self.clip_grad_norm <= 0:
            raise ValueError(
                f"clip_grad_norm should be positive, got {self.clip_grad_norm}."
            )
        self.gather_norm = False
        self.grad_norms = []
        self.clip_coef = None

        self.grad_func = self.fuse_update()
        self.first_backward = True  # check bf16 or fp16 in the first backward

        # register hook function, which will be called through the backward process
        for n, p in self.named_params:
            if p.requires_grad:
                p.register_hook(self.grad_func)
        defaults = dict(
            lr=lr, clip_grad_norm=clip_grad_norm, clip_grad_value=clip_grad_value
        )
        super().__init__([p for _, p in self.named_params], defaults)

    def fuse_update(self):
        def func(x):
            with torch.no_grad():
                for n, p in self.named_params:
                    if p.requires_grad and p.grad is not None:
                        if self.loss_scaler and (
                            self.loss_scaler.has_overflow_serial
                            or self.loss_scaler._has_inf_or_nan(p.grad)
                        ):
                            # if the overflow is detected, drop the gradient
                            p.grad = None
                            self.loss_scaler.has_overflow_serial = True
                            break
                        grad_fp32 = p.grad.to(torch.float32)
                        p.grad = None
                        if self.loss_scaler:
                            grad_fp32.div_(self.loss_scaler.loss_scale)
                        if self.gather_norm:
                            # we adopt two backward pass for gradient norm computation and parameter update, respectively.
                            self.grad_norms.append(torch.norm(grad_fp32, 2.0))
                        else:
                            if (
                                self.clip_grad_value is not None
                                and self.clip_grad_value > 0
                            ):
                                # Clipping gradients by their value
                                grad_fp32.clamp_(
                                    min=-self.clip_grad_value, max=self.clip_grad_value
                                )
                            if (
                                self.clip_grad_norm is not None
                                and self.clip_grad_norm > 0
                                and self.clip_coef is not None
                            ):
                                # Normalize the gradient according to its norm (computed in another pass)
                                grad_fp32.mul_(self.clip_coef)
                            p_fp32 = p.data.to(torch.float32)
                            if self.do_weight_decay:
                                p_fp32.mul_(1.0 - self.lr * self.weight_decay)
                            p_fp32.add_(grad_fp32, alpha=-self.lr)
                            p.data.copy_(p_fp32)

            return x

        return func

    def fused_backward(self, loss, lr):
        if self.first_backward:
            self.first_backward = False
            if loss.dtype == torch.float16:
                self.loss_scaler = DynamicLossScaler(**self.loss_scale_args)
                if self.clip_grad_norm is None:
                    self.clip_grad_norm = 1.0
                    warnings.warn(
                        "Loss scale is recommended to be used with grad norm to get better performance. "
                        "Set grad norm to 1.0."
                    )
        self.lr = lr
        # Users need call grad_norm themselves and then call backward_step
        if (
            self.clip_grad_norm is not None
            and self.clip_grad_norm > 0
            and self.clip_coef is None
        ):
            raise ValueError(
                "clip_grad_norm is not None, but clip_coef is None. "
                "Please call optimizer.grad_norm() before optimizer.fused_backward()."
            )
        if self.loss_scaler:
            loss = loss * self.loss_scaler.loss_scale
        loss.backward()
        # update the last parameter since the last parameter in the computation graph is not ready when calling hook functions
        # the argument of grad_func is just a placeholder, and it can be anything.
        self.grad_func(0)

    def grad_norm(self, loss):
        if self.first_backward:
            self.first_backward = False
            if loss.dtype == torch.float16:
                self.loss_scaler = DynamicLossScaler(**self.loss_scale_args)

        self.gather_norm = True
        self.grad_norms = []
        if self.loss_scaler:
            self.loss_scaler.has_overflow_serial = False
            loss = loss * self.loss_scaler.loss_scale
        loss.backward(retain_graph=True)
        # update the last parameter since the last parameter in the computation graph is not ready when calling hook functions
        # the argument of grad_func is just a placeholder, and it can be anything.
        self.grad_func(0)

        if self.loss_scaler and self.loss_scaler.has_overflow_serial:
            self.loss_scaler.update_scale(overflow=True)
            with torch.no_grad():  # clear gradients
                for n, p in self.named_params:
                    p.grad = None
            return

        with torch.no_grad():
            # The norm is computed over all gradients together, as if they were
            # concatenated into a single vector. Gradients are modified in-place.
            self.grad_norms = torch.stack(self.grad_norms)

            total_norm = torch.norm(self.grad_norms, 2.0)
            self.clip_coef = float(self.clip_grad_norm) / (total_norm + 1e-6)
            self.clip_coef = torch.clamp(self.clip_coef, max=1.0)
        self.gather_norm = False


class AdaLomo(Optimizer):
    r"""Implements AdaLomo, low-memory optimization with adaptive learning rates.

    .. math::
       \begin{aligned}
       \beta_{2,t} &= 1 - t^{c} \\
       R_t &= \beta_{2,t} R_{t-1}
              + (1 - \beta_{2,t})\,(G_t^{\,2} + \epsilon_1 1_n 1_m^\top)\, 1_m \\
       C_t &= \beta_{2,t} C_{t-1}
              + (1 - \beta_{2,t})\, 1_n^\top (G_t^{\,2} + \epsilon_1 1_n 1_m^\top) \\
       \hat{V}_t &= R_t C_t / (1_n^\top R_t) \\
       U_t &= G_t / \sqrt{\hat{V}_t}, \qquad
       \hat{U}_t = U_t / \max\!\bigl(1, \mathrm{RMS}(U_t)/d\bigr) \\
       \theta_t &= (1 - \eta_t \lambda)\,\theta_{t-1} - \eta_t\, \hat{U}_t,
       \qquad \eta_t = \eta \max\!\bigl(\epsilon_2, \mathrm{RMS}(\theta_{t-1})\bigr)
       \end{aligned}

    for a matrix parameter :math:`\theta \in \mathbb{R}^{n \times m}` with
    gradient :math:`G_t`, where :math:`c` is ``decay_rate`` (default
    :math:`-0.8`), :math:`(\epsilon_1, \epsilon_2)` is ``eps``, :math:`d` is
    ``clip_threshold``, and :math:`\lambda` is the decoupled
    ``weight_decay``. Vector parameters keep an unfactored second moment.
    As in LOMO, the update is computed and applied inside the backward pass,
    so only the factored second moment of Adafactor (Shazeer and Stern,
    ICML 2018) persists between steps.

    Reference: Kai Lv, Hang Yan, Qipeng Guo, Haijun Lv, Xipeng Qiu,
    "AdaLomo: Low-memory Optimization with Adaptive Learning Rate",
    Findings of ACL 2024.
    https://arxiv.org/abs/2310.10195

    Note:
        Drive training with :meth:`fused_backward` instead of
        ``loss.backward()`` followed by ``step()``. When ``clip_grad_norm``
        is set, call :meth:`grad_norm` on the loss first. The second-moment
        buffers live outside ``Optimizer.state`` and are not captured by
        :meth:`state_dict`.
    """

    def __init__(
        self,
        model,
        lr=1e-3,
        loss_scale=2**10,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        clip_grad_norm=None,
        clip_grad_value=None,
        weight_decay=0.0,
    ):
        self.named_params = _named_parameters(model)
        self.lr = lr
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value
        self.weight_decay = weight_decay
        self.loss_scale = loss_scale
        if self.weight_decay > 0.0:
            self.do_weight_decay = True
        else:
            self.do_weight_decay = False
        self.eps = eps
        self.step_num = 0
        self.decay_rate = decay_rate
        self.clip_threshold = clip_threshold

        # for grad norm
        if self.clip_grad_norm is not None and self.clip_grad_norm <= 0:
            raise ValueError(
                f"clip_grad_norm should be positive, got {self.clip_grad_norm}."
            )
        self.gather_norm = False
        self.grad_norms = []
        self.clip_coef = None

        self.grad_func = self.fuse_update()

        self.exp_avg_sq = {}
        self.exp_avg_sq_row = {}
        self.exp_avg_sq_col = {}

        # register hook function, which will be called through the backward process
        for n, p in self.named_params:
            if len(p.data.shape) == 1:
                self.exp_avg_sq[n] = torch.zeros(
                    p.data.shape[0], dtype=torch.float32, device=p.device
                )
            else:
                self.exp_avg_sq_row[n] = torch.zeros(
                    p.data.shape[0], dtype=torch.float32, device=p.device
                )
                self.exp_avg_sq_col[n] = torch.zeros(
                    p.data.shape[1], dtype=torch.float32, device=p.device
                )

            if p.requires_grad:
                p.register_hook(self.grad_func)
        defaults = dict(
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            clip_grad_norm=clip_grad_norm,
            clip_grad_value=clip_grad_value,
        )
        super().__init__([p for _, p in self.named_params], defaults)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        # copy from fairseq's adafactor implementation:
        # https://github.com/huggingface/transformers/blob/8395f14de6068012787d83989c3627c3df6a252b/src/transformers/optimization.py#L505
        r_factor = (
            (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
            .rsqrt_()
            .unsqueeze(-1)
        )
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def fuse_update(self):
        def func(x):
            with torch.no_grad():
                for n, p in self.named_params:
                    if p.requires_grad and p.grad is not None:
                        grad_fp32 = p.grad.to(torch.float32)
                        p.grad = None
                        if self.loss_scale:
                            grad_fp32.div_(self.loss_scale)
                        if self.gather_norm:
                            # we adopt two backward pass for gradient norm computation and parameter update, respectively.
                            self.grad_norms.append(torch.norm(grad_fp32, 2.0))
                        else:
                            # grad clip or norm
                            if (
                                self.clip_grad_value is not None
                                and self.clip_grad_value > 0
                            ):
                                # Clipping gradients by their value
                                grad_fp32.clamp_(
                                    min=-self.clip_grad_value, max=self.clip_grad_value
                                )
                            if (
                                self.clip_grad_norm is not None
                                and self.clip_grad_norm > 0
                                and self.clip_coef is not None
                            ):
                                # Normalize the gradient according to its norm (computed in another pass)
                                grad_fp32.mul_(self.clip_coef)

                            # To avoid math errors for edge cases
                            if self.step_num == 0 and self.decay_rate < 0:
                                decay_rate = -self.decay_rate
                            else:
                                decay_rate = self.decay_rate

                            beta2t = 1.0 - math.pow(self.step_num, decay_rate)
                            update = (grad_fp32**2) + self.eps[0]

                            if len(p.data.shape) > 1:
                                self.exp_avg_sq_row[n].mul_(beta2t).add_(
                                    update.mean(dim=-1), alpha=1.0 - beta2t
                                )
                                self.exp_avg_sq_col[n].mul_(beta2t).add_(
                                    update.mean(dim=-2), alpha=1.0 - beta2t
                                )
                                update = self._approx_sq_grad(
                                    self.exp_avg_sq_row[n], self.exp_avg_sq_col[n]
                                )
                                update.mul_(grad_fp32)
                            else:
                                self.exp_avg_sq[n].mul_(beta2t).add_(
                                    update, alpha=1.0 - beta2t
                                )
                                update = self.exp_avg_sq[n].rsqrt().mul_(grad_fp32)

                            update.div_(
                                (self._rms(update) / self.clip_threshold).clamp_(
                                    min=1.0
                                )
                            )

                            p_fp32 = p.data.to(torch.float32)
                            p_rms = torch.norm(p_fp32, 2.0) / math.sqrt(p.numel())
                            lr = self.lr
                            param_scale = max(self.eps[1], p_rms)
                            lr = lr * param_scale

                            if self.do_weight_decay:
                                p_fp32.mul_(1.0 - lr * self.weight_decay)
                            p_fp32.add_(update, alpha=-lr)
                            p.data.copy_(p_fp32)

            return x

        return func

    def fused_backward(self, loss, lr):
        self.lr = lr
        if self.loss_scale:
            loss = loss * self.loss_scale
        self.step_num += 1
        loss.backward()
        # update the last parameter since the last parameter in the computation graph is not ready when calling hook functions
        # the argument of grad_func is just a placeholder, and it can be anything.
        self.grad_func(0)

    def grad_norm(self, loss):
        self.gather_norm = True
        self.grad_norms = []
        if self.loss_scale:
            loss = loss * self.loss_scale
        loss.backward(retain_graph=True)
        # update the last parameter since the last parameter in the computation graph is not ready when calling hook functions
        # the argument of grad_func is just a placeholder, and it can be anything.
        self.grad_func(0)

        with torch.no_grad():
            # The norm is computed over all gradients together, as if they were
            # concatenated into a single vector. Gradients are modified in-place.
            self.grad_norms = torch.stack(self.grad_norms)

            total_norm = torch.norm(self.grad_norms, 2.0)
            self.clip_coef = float(self.clip_grad_norm) / (total_norm + 1e-6)
            self.clip_coef = torch.clamp(self.clip_coef, max=1.0)
        self.gather_norm = False


class DynamicLossScaler:
    """Dynamic loss scaler for fp16 training, as used by LOMO."""

    def __init__(
        self,
        init_scale=2**32,
        scale_factor=2.0,
        scale_window=1000,
        min_scale=1,
        delayed_shift=1,
        consecutive_hysteresis=False,
        raise_error_at_min_scale=True,
        dtype=torch.half,
    ):
        self.cur_scale = init_scale
        self.cur_iter = 0
        self.last_overflow_iter = -1
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.min_scale = min_scale
        self.delayed_shift = delayed_shift
        self.cur_hysteresis = delayed_shift
        self.consecutive_hysteresis = consecutive_hysteresis
        self.raise_error_at_min_scale = raise_error_at_min_scale
        self.dtype = dtype
        self.has_overflow_serial = False

    @property
    def loss_scale(self):
        return self.cur_scale

    # `x` is a torch.Tensor
    def _has_inf_or_nan(self, x):
        try:
            # if x is half, the .float() incurs an additional deep copy, but it's necessary if
            # Pytorch's .sum() creates a one-element tensor of the same type as x
            # (which is true for some recent version of pytorch).
            cpu_sum = float(x.float().sum())
            # More efficient version that can be used if .sum() returns a Python scalar
            # cpu_sum = float(x.sum())
        except RuntimeError as instance:
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum in [float("inf"), -float("inf")] or cpu_sum != cpu_sum:
                return True
            return False

    # `overflow` is boolean indicating whether the gradient overflowed
    def update_scale(self, overflow):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if overflow:
            # self.cur_scale /= self.scale_factor
            if self.delayed_shift == 1 or self.cur_hysteresis == 1:
                if (self.cur_scale == self.min_scale) and self.raise_error_at_min_scale:
                    raise Exception(
                        "Current loss scale already at minimum - cannot decrease scale anymore. Exiting run."
                    )
                else:
                    next_scale = max(self.cur_scale / self.scale_factor, self.min_scale)
                    if rank == 0:
                        overflow_msg = f"[LOMO] OVERFLOW! Rank {rank} Skipping step."
                        if self.dtype == torch.half:
                            overflow_msg += f" Attempted loss scale: {int(self.cur_scale)}, reducing to {int(next_scale)}"
                        print(overflow_msg)
                    self.cur_scale = next_scale
            else:
                if rank == 0:
                    overflow_msg = f"[LOMO] OVERFLOW! Rank {rank} Skipping step."
                    if self.dtype == torch.half:
                        overflow_msg += f" Attempted loss scale: {int(self.cur_scale)}, but hysteresis is {self.cur_hysteresis}. Reducing hysteresis to {self.cur_hysteresis - 1}"
                    print(overflow_msg)
                self.cur_hysteresis -= 1
            self.last_overflow_iter = self.cur_iter
        else:
            if self.consecutive_hysteresis:
                if rank == 0:
                    hysteresis_msg = f"Consecutive hysteresis is enabled. Restoring hysteresis to {self.delayed_shift}"
                    print(hysteresis_msg)
                self.cur_hysteresis = self.delayed_shift
            if (self.cur_iter - self.last_overflow_iter) % self.scale_window == 0:
                if not self.consecutive_hysteresis:
                    self.cur_hysteresis = self.delayed_shift
                self.cur_scale *= self.scale_factor
        self.cur_iter += 1
