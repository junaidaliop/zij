# Adapted from https://github.com/Ledzy/BAdam (commit bf989dc)
# Copyright (c) 2024 Qijun Luo. Licensed under the Apache-2.0 license.
# See THIRD_PARTY_NOTICES.md for details.
"""Implementation of the BAdam block coordinate optimizers."""

import gc
import math
import re
import warnings
from collections import defaultdict
from typing import Dict, List, Optional

import torch
from torch import Tensor

from ...core.adam import Adam
from ...core.optimizer import Optimizer

__all__ = ["BlockOptimizer", "BlockOptimizerRatio"]


def print_rank_0(s, force=True):
    if not torch.distributed.is_initialized():
        print(s)
    elif torch.distributed.get_rank() == 0 and force:
        print(s)


def _named_parameters(params):
    params = list(params)
    if params and isinstance(params[0], Tensor):
        return [(f"param_{i}.", p) for i, p in enumerate(params)]
    return [(n, p) for n, p in params]


class BlockOptimizer(Optimizer):
    r"""Implements BAdam, block coordinate descent with Adam as the inner solver.

    The parameters are partitioned into :math:`D` blocks
    :math:`\theta = (\theta_{\pi_1}, \ldots, \theta_{\pi_D})`. Only the
    active block :math:`\pi_i` is trainable; it receives :math:`K` steps of
    the base optimizer, Adam in the paper, before the next block becomes
    active and the optimizer state is reset:

    .. math::
       \begin{aligned}
       g_k &= \nabla_{\theta_{\pi_i}} \mathcal{L}(\theta) \\
       m_k &= \beta_1 m_{k-1} + (1 - \beta_1) g_k \\
       v_k &= \beta_2 v_{k-1} + (1 - \beta_2) g_k^2 \\
       \theta_{\pi_i} &\leftarrow \theta_{\pi_i} - \eta\,
           \frac{m_k / (1 - \beta_1^k)}{\sqrt{v_k / (1 - \beta_2^k)}
           + \epsilon}
       \end{aligned}

    with :math:`m_0 = v_0 = 0` at every block switch and all blocks other
    than :math:`\pi_i` frozen. Only the active block carries optimizer state
    and a float32 master copy, so the memory overhead is that of a single
    block rather than the full model.

    Reference: Qijun Luo, Hengxu Yu, Xiao Li,
    "BAdam: A Memory Efficient Full Parameter Optimization Method for Large
    Language Models", NeurIPS 2024.
    https://arxiv.org/abs/2404.02827

    Note:
        Pass ``model.named_parameters()`` as ``params`` so blocks can be
        inferred from transformer layer names, or set ``block_prefix_list``
        explicitly; a plain parameter list falls back to one block per
        parameter. ``base_optimizer`` may be an optimizer class, constructed
        with the remaining keyword arguments, or an already constructed
        instance. The memory savings assume fp16/bf16 model weights.
    """

    def __init__(
        self,
        params,
        base_optimizer=Adam,
        block_prefix_list: Optional[List[List[str]]] = None,
        switch_block_every: int = 50,
        start_block: Optional[int] = None,
        switch_mode: str = "descending",
        active_modules: List[str] = [],
        include_embedding=False,
        include_lm_head=False,
        verbose: int = 1,
        log_fn=None,
        **kwargs,
    ):
        named_parameters_list = _named_parameters(params)
        if isinstance(base_optimizer, type):
            base_optimizer = base_optimizer(
                [p for _, p in named_parameters_list], **kwargs
            )

        if block_prefix_list is None:
            block_prefix_list = self.infer_param_groups(
                [n for n, _ in named_parameters_list],
                include_embedding,
                include_lm_head,
            )
            if not block_prefix_list:
                block_prefix_list = [[n] for n, _ in named_parameters_list]

        assert switch_mode in ["random", "descending", "ascending", "fixed"]
        assert isinstance(block_prefix_list, list)

        self.verbose = verbose
        self.switch_mode = switch_mode
        self.switch_block_every = switch_block_every
        self.named_parameters_list = named_parameters_list
        self.weight_decay = base_optimizer.param_groups[0]["weight_decay"]
        self.block_prefix_list = block_prefix_list
        self.block_num = len(block_prefix_list)
        self.log_fn = log_fn
        self.global_step = 0
        self.base_optimizer = base_optimizer
        self.active_modules = active_modules
        self.defaults = base_optimizer.defaults

        self.param_groups = base_optimizer.param_groups

        if start_block is not None:
            self.current_block_idx = start_block
        elif self.switch_mode == "descending":
            self.current_block_idx = self.block_num - 1
        elif self.switch_mode == "ascending":
            self.current_block_idx = 0
        elif self.switch_mode == "random":
            self.block_order = torch.randperm(self.block_num).tolist()
            print_rank_0(
                f"next block epoch's update order: {self.block_order[::-1]}"
            )
            self.current_block_idx = self.block_order.pop()

        # detect if in lora mode or not
        self.lora_mode = False
        if any("lora" in n for n, _ in named_parameters_list):
            self.lora_mode = True
            print_rank_0("LoRA mode detected. Will only train the lora parameters.")

        fp32_params = []
        for n, p in named_parameters_list:
            if p.dtype == torch.float32:
                fp32_params.append(n)
        if len(fp32_params) > 0:
            warnings.warn(
                f"BAdam expect model to be loaded in fp16/bf16 precision, while detect fp32 "
                f"weight for the following parameters: {fp32_params} \n"
                "This will cause additional memory usage and lose the benefit of mixed precision training."
            )

        super().__init__(self.param_groups, base_optimizer.defaults)

        self.switch_trainable_params()

    @property
    def embedding_layer(self):
        for n, p in self.named_parameters_list:
            if "embed" in n:
                return p

    @property
    def lm_head_layer(self):
        for n, p in self.named_parameters_list:
            if "lm_head" in n:
                return p

    def infer_param_groups(self, param_names, include_embedding, include_lm_head):
        """automatic inference of the parameter groups based on the parameter names.
        divide groups into:
            * embedding
            * transformer layers
            * lm_head and others
        """
        block_prefix_list = []
        lm_head_and_other_params = []
        embed_pattern = r'.*embed[^.]*\.'
        layer_pattern = r'.*layers.[^.]*\.'

        for name in param_names:
            if any(prefix[0] in name for prefix in block_prefix_list):
                continue

            if re.findall(layer_pattern, name):
                block_prefix_list.append(re.findall(layer_pattern, name))
            elif re.findall(embed_pattern, name) and include_embedding:
                block_prefix_list.append(re.findall(embed_pattern, name))
            else:
                lm_head_and_other_params.append(name)

        if include_lm_head:
            block_prefix_list.append(lm_head_and_other_params)

        return block_prefix_list

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        return self.base_optimizer.load_state_dict(state_dict)

    def _update_lr(self):
        # Make sure the learning rate of the base_optimizer is consistent with the BlockOptimizer
        for group in self.base_optimizer.param_groups:
            group["lr"] = self.param_groups[0]["lr"]

    def step(self, *args, **kwargs) -> None:
        self._update_lr()
        self._grad_to_hp()
        self.base_optimizer.step(*args, **kwargs)
        self._update_param()
        self._clean_hp_grad()

        self.global_step += 1

        torch.cuda.empty_cache()

        if (self.global_step + 1) % self.switch_block_every == 0:
            self.switch_trainable_params()

    def _clean_hp_grad(self) -> None:
        """Clean the gradients of the high precision parameters."""
        for hp_param in self.param_idx2hp.values():
            hp_param.grad = None

    def _update_param(self) -> None:
        """Update the low precision parameters with the values of the high precision parameters."""
        for lp_param, hp_param in zip(self.param_idx2lp.values(), self.param_idx2hp.values()):
            lp_param.data.copy_(hp_param.to(lp_param.dtype).data)

    def _grad_to_hp(self, clear_lp_grads: bool = True) -> None:
        """Convert the gradients of the low precision parameters to high precision."""
        for lp_param, hp_param in zip(self.param_idx2lp.values(), self.param_idx2hp.values()):
            assert lp_param.grad is not None, "The low precision parameter's gradient is None."
            hp_param.grad = lp_param.grad.float()

            if clear_lp_grads:
                lp_param.grad = None

    def switch_trainable_params(self, verbose: Optional[int] = None) -> None:
        """Update the trainable parameters based on the current block index."""
        if verbose is None:
            verbose = self.verbose

        self.active_param_prefixs = self.block_prefix_list[self.current_block_idx] + self.active_modules

        # Make sure there are trainable parameters in the current block when using lora
        while self.lora_mode:
            active_param_names = [n for n, _ in self.named_parameters_list if any(p in n for p in self.active_param_prefixs)]
            if all("lora" not in n for n in active_param_names):
                print_rank_0(f"In LoRA mode but no LoRA parameters in the current block with prefix: {self.active_param_prefixs}. Switching to the next block.")
                self._update_active_block_idx()
                self.active_param_prefixs = self.block_prefix_list[self.current_block_idx] + self.active_modules
                continue
            break

        if verbose >= 1:
            print_rank_0(f"Parameters with the following prefix will be trainable: {self.active_param_prefixs}")

        self._switch_trainable_params_single_gpu(verbose)

        # Clean the optimizer state
        self.base_optimizer.state = defaultdict(lambda: {})
        self._update_active_block_idx()
        gc.collect()

    def _switch_trainable_params_single_gpu(self, verbose: int) -> None:
        self.param_idx2lp = {}
        self.param_idx2hp = {}

        active_param_groups = [
            {
                "params": [],
                "weight_decay": self.param_groups[0]['weight_decay'],
                **self.defaults
            },
            {
                "params": [],
                "weight_decay": 0.0,
                **self.defaults
            },
        ]

        for i, (name, param) in enumerate(self.named_parameters_list):
            if not any(p in name for p in self.active_param_prefixs):
                param.requires_grad_(False)
                param.grad = None
            else:
                if self.lora_mode and "lora" not in name:
                    continue
                param.requires_grad_(True)
                param_hp = param.clone().float().detach().to(param.device)
                param_hp.requires_grad = True

                self.param_idx2lp[i] = param
                self.param_idx2hp[i] = param_hp

                if "bias" not in name:
                    active_param_groups[0]['params'].append(param_hp)
                else:
                    active_param_groups[1]['params'].append(param_hp)

                if verbose >= 2:
                    print_rank_0(name)
        self.base_optimizer.param_groups = active_param_groups

    def _update_active_block_idx(self):
        # Update the trainable block
        if self.switch_mode == "random":
            if len(self.block_order) == 0:
                self.block_order = torch.randperm(self.block_num).tolist()
                print_rank_0(
                    f"Next block epoch's update order: {self.block_order[::-1]}"
                )
            self.current_block_idx = self.block_order.pop()
        elif self.switch_mode == "ascending":
            self.current_block_idx = (self.current_block_idx + 1) % self.block_num
        elif self.switch_mode == "descending":
            self.current_block_idx = (self.current_block_idx - 1) % self.block_num
        elif self.switch_mode == "fixed":
            pass


class BlockOptimizerRatio(Optimizer):
    r"""Implements the ratio-based variant of BAdam, where every active block holds a fixed fraction of each parameter's entries.

    Instead of cycling through structural blocks such as transformer layers,
    each active block selects ``update_ratio`` of the coordinates of every
    parameter, either adjacent in memory or randomly scattered, and applies
    the Adam update restricted to that coordinate set for ``switch_every``
    steps before moving to the next set:

    .. math::
       \begin{aligned}
       m_k &= \beta_1 m_{k-1} + (1 - \beta_1) g_k \\
       v_k &= \beta_2 v_{k-1} + (1 - \beta_2) g_k^2 \\
       \theta &\leftarrow \theta - \eta\,
           \frac{\sqrt{1 - \beta_2^k}}{1 - \beta_1^k}
           \frac{m_k}{\sqrt{v_k} + \epsilon}
       \end{aligned}

    where :math:`g_k` is the gradient masked to the active coordinates and
    the optimizer state is stored sparsely for those coordinates only, then
    reset at every switch.

    Reference: Qijun Luo, Hengxu Yu, Xiao Li,
    "BAdam: A Memory Efficient Full Parameter Optimization Method for Large
    Language Models", NeurIPS 2024.
    https://arxiv.org/abs/2404.02827

    Note:
        Gradients are sparsified through a post-accumulate-grad hook on each
        trainable parameter, so plain ``loss.backward()`` followed by
        ``step()`` works unmodified. Parameters whose names contain
        ``embed`` or ``lm_head`` are frozen unless ``include_embedding`` or
        ``include_lm_head`` is set.
    """

    def __init__(self, params,
                 update_ratio=0.1,
                 verbose=1,
                 switch_every=100,
                 preserve_threshold=100,
                 param_update_ratios=defaultdict(lambda: None),
                 mask_mode="adjacent",
                 lr=1e-5,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 optimizer_defaults=None,
                 keep_mask=True,
                 include_embedding=False,
                 include_lm_head=False
                 ):
        named_parameters_list = _named_parameters(params)
        self.update_ratio = update_ratio
        self.verbose = verbose
        self.sparse_hook = self.sparse_update_hook()
        self.param_groups = [p for _, p in named_parameters_list]
        self.named_parameters_list = named_parameters_list
        self.sparse_dict = defaultdict(lambda: {})
        self.switch_every = switch_every
        self.preserve_threshold = preserve_threshold
        self.global_step = 0

        if not include_embedding and self.embedding_layer is not None:
            self.embedding_layer.requires_grad_(False)
        if not include_lm_head and self.lm_head_layer is not None:
            self.lm_head_layer.requires_grad_(False)

        # mask
        self.mask_mode = mask_mode
        self.keep_mask = keep_mask
        self.mask_dict = {}

        for n, p in named_parameters_list:
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self.sparse_hook)
            self.sparse_dict[p]["offset"] = 0
            self.sparse_dict[p]["seed"] = torch.randint(0, 1000, (1,)).item() # seed for each parameter's random index generator

            for param_name_prefix in param_update_ratios.keys():
                if param_name_prefix in n:
                    self.sparse_dict[p]["update_ratio"] = param_update_ratios[param_name_prefix]
                    continue

        defaults = dict(lr=lr, betas=betas, eps=eps) if optimizer_defaults is None else optimizer_defaults
        super().__init__(self.param_groups, defaults)

    @property
    def embedding_layer(self):
        for n, p in self.named_parameters_list:
            if "embed" in n:
                return p

    @property
    def lm_head_layer(self):
        for n, p in self.named_parameters_list:
            if "lm_head" in n:
                return p

    def _sparse_adam(self,
                     params: List[Tensor],
                     grads: List[Tensor],
                     exp_avgs: List[Tensor],
                     exp_avg_sqs: List[Tensor],
                     state_steps: List[int],
                     *,
                     eps: float,
                     beta1: float,
                     beta2: float,
                     lr: float,
                     maximize: bool):
        """Functional API that performs the sparse Adam computation on the active coordinates."""
        for i, param in enumerate(params):
            grad = grads[i]
            grad = grad if not maximize else -grad
            grad = grad.coalesce()  # the update is non-linear so indices must be unique
            grad_indices = grad._indices()
            grad_values = grad._values()
            size = grad.size()

            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]

            def make_sparse(values):
                constructor = grad.new
                if grad_indices.dim() == 0 or values.dim() == 0:
                    return constructor().resize_as_(grad)
                return constructor(grad_indices, values, size)

            # Decay the first and second moment running average coefficient
            #      old <- b * old + (1 - b) * new
            # <==> old += (1 - b) * (new - old)
            exp_avg_update = grad_values.sub(exp_avg).mul_(1 - beta1)
            exp_avg.add_(exp_avg_update)
            exp_avg_sq_update = grad_values.pow(2).sub_(exp_avg_sq).mul_(1 - beta2)
            exp_avg_sq.add_(exp_avg_sq_update)

            # Dense addition again is intended, avoiding another sparse_mask
            numer = exp_avg.clone()
            denom = exp_avg_sq.clone().sqrt_().add_(eps)
            del exp_avg_update, exp_avg_sq_update

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1

            update_direction = make_sparse(numer.div_(denom))
            param.add_(-step_size * update_direction)

            del update_direction

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']
            maximize = group.get('maximize', False)

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if not p.grad.is_sparse:
                        raise RuntimeError('SparseAdam does not support dense gradients, please consider Adam instead')
                    grads.append(p.grad)

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # NOTE: the exp_avg is a vector instead of matrix, since we only store the states for the non-zero entries
                        state['exp_avg'] = torch.zeros_like(p.grad._values())

                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p.grad._values())

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            self._sparse_adam(params_with_grad,
                              grads,
                              exp_avgs,
                              exp_avg_sqs,
                              state_steps,
                              beta1=beta1,
                              beta2=beta2,
                              lr=group['lr'],
                              eps=group['eps'],
                              maximize=maximize)

        self.global_step += 1
        torch.cuda.empty_cache()

        if self.global_step % self.switch_every == 0:
            self._reset_state_dict()

        return loss

    def _reset_state_dict(self):
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p] = defaultdict()

    def _generate_mask_adjacent(self, param, ratio, offset):
        """select a group of adjacent entries in the matrix, starting from the offset. If the end of the matrix is reached, continue from the beginning."""
        num_elements = param.numel()
        num_ones = int(num_elements * ratio)

        if offset + num_ones > num_elements:
            i1 = torch.arange(0, offset + num_ones - num_elements, device=param.device).unsqueeze(0)
            i2 = torch.arange(offset, num_elements, device=param.device).unsqueeze(0)
            i = torch.cat([i1, i2], dim=1)
        else:
            i = torch.arange(offset, min(offset + num_ones, num_elements), device=param.device).unsqueeze(0)
        unraveled_i = torch.vstack(torch.unravel_index(i, param.size()))
        mask = torch.sparse_coo_tensor(unraveled_i, torch.ones(num_ones, device=param.device, dtype=param.dtype), param.shape)

        return mask

    def _generate_mask_scatter(self, param, ratio, offset):
        """randomly select entries in the matrix. The selected entries are not necessarily adjacent.
        The indices are recorded by setting the seed.
        """
        num_elements = param.numel()
        num_ones = int(num_elements * ratio)

        torch.random.manual_seed(self.sparse_dict[param]["seed"])
        randperm = torch.randperm(num_elements, device=param.device)
        if offset + num_ones > num_elements:
            i1 = randperm[offset:]
            i2 = randperm[:offset + num_ones - num_elements]
            i = torch.cat([i1, i2])
        else:
            i = randperm[offset:offset+num_ones]

        unraveled_i = torch.vstack(torch.unravel_index(i, param.size()))
        mask = torch.sparse_coo_tensor(unraveled_i, torch.ones(num_ones, device=param.device, dtype=param.dtype), param.shape)

        return mask

    def sparse_update_hook(self):

        def func(p):
            num_elements = p.numel()
            offset = self.sparse_dict[p]["offset"]
            update_ratio = self.sparse_dict[p]["update_ratio"] if "update_ratio" in self.sparse_dict[p] else self.update_ratio

            # when the parameter is too small, we simply sparsify the whole gradient
            if num_elements < self.preserve_threshold:
                p.grad = p.grad.add_(1e-9).to_sparse()

            elif update_ratio == 1.:
                p.grad = p.grad.add_(1e-9).to_sparse()
            else:
                if p.shape in self.mask_dict and self.mask_dict[p.shape] is not None:
                    mask = self.mask_dict[p.shape]
                else:
                    if self.mask_mode == "adjacent":
                        mask = self._generate_mask_adjacent(p, update_ratio, offset)
                    elif self.mask_mode == "scatter":
                        mask = self._generate_mask_scatter(p, update_ratio, offset)
                    else:
                        raise NotImplementedError

                    # We save the same mask for all the parameters with the same shape, this treats memory for time.
                    if self.keep_mask:
                        self.mask_dict[p.shape] = mask

                p.grad = p.grad.sparse_mask(mask)

                if (self.global_step + 1) % self.switch_every == 0:
                    self.sparse_dict[p]["offset"] = (offset + int(num_elements * update_ratio)) % num_elements
                    self.mask_dict[p.shape] = None

        return func
