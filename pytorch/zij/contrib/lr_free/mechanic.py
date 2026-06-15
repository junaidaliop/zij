# Adapted from https://github.com/optimizedlearning/mechanic (commit 4a037bc)
# Copyright (c) 2023 Ashok Cutkosky. Licensed under the MIT license.
# See THIRD_PARTY_NOTICES.md for details.
"""Implementation of the Mechanic learning rate tuner."""

import logging
from typing import Any, Callable, Dict, Tuple

import torch

__all__ = ["Mechanic", "is_mechanized", "mechanize"]


def _init_state(
    optimizer: torch.optim.Optimizer,
    p_ref: Dict[torch.Tensor, torch.Tensor],
    s_decay: float,
    betas: Tuple[float],
    s_init: float,
    eps: float,
    store_delta: bool,
    log_every: int,
    force=False,
):
    """Initialize extra state for mechanic.

    Args:
        optimizer: optimizer instance to initialize extra state for.
        p_ref: mapping of parameters to their initial values at the start of optimization.
        s_decay: how much "weight decay" analog to add (called lambda in the paper).
        betas: list of beta values.
        s_init: initial scale value.
        eps: small number for numerical precision.
        store_delta: whether to store the offsets or recompute them on-the-fly.
        log_every: how often to log scale values.
        force: if True, reinitialize the state.
    """
    if force or "_mechanic" not in optimizer.state:
        optimizer.state["_mechanic"] = {
            "s_decay": torch.tensor(s_decay),
            "betas": torch.tensor(betas),
            "s_init": torch.tensor(s_init),
            "eps": eps,
            "s": torch.zeros(len(betas)),
            "p_ref": {},
            "sum_squared_products": torch.zeros(len(betas)),
            "reward": torch.zeros(len(betas)),
            "max_product": torch.full((len(betas),), 1e-6),
            "iter_count": 0,
            "log_every": log_every,
        }
        _init_reference(optimizer, p_ref, store_delta)


def _init_reference(
    optimizer: torch.optim.Optimizer,
    p_ref: Dict[torch.Tensor, torch.Tensor],
    store_delta: bool,
):
    """Store the starting point of the optimization (the "reference").

    Args:
        optimizer: optimizer instance to store reference for.
        p_ref: mapping of parameters to their initial values at the start of optimization.
        store_delta: if true, we should also store the "Delta" value: the
            displacement between the current iterate and the reference.
    """
    for group in optimizer.param_groups:
        for p in group["params"]:
            optimizer.state["_mechanic"][p] = {
                "ref": p_ref[p].clone(),
            }
            if store_delta:
                optimizer.state["_mechanic"][p]["delta"] = torch.zeros_like(p)


def _step(
    optimizer: torch.optim.Optimizer,
    base_step: Callable,
    s_decay: float,
    betas: Tuple[float],
    s_init: float,
    eps: float,
    store_delta: bool = True,
    log_every: int = 0,
    closure: Callable = None,
):
    """Run one step of mechanic.

    Args:
        optimizer: mechanic optimizer instance that we are computing the step for.
        base_step: The "step" function of the base optimizer (e.g. SGD, AdamW etc).
        s_decay: how much "weight decay" analog to add (called lambda in the paper).
        betas: list of beta values.
        s_init: initial scale value.
        eps: small number for numerical precision.
        store_delta: whether to store the offsets between current iterate and reference
            or recompute them on-the-fly.
        closure: closure that reevaluates the model and returns the loss.
    Returns:
        loss value
    """
    prev_grad = torch.is_grad_enabled()

    # we don't wrap the entire function in @torch.no_grad because
    # we want to let the base optimizer differentiate things
    # if it so desires.
    torch.set_grad_enabled(False)

    if closure is not None:
        # if we need to rely on closure to generate gradients
        # then we generate gradient here, but also need to let the
        # base algorithm potentially reevaluate the closure as much
        # as it likes without doubling the gradients the first time it does so.
        # So, we will create a "fake" closure called skip_once_closure
        # to be eventually provided to base_step.
        loss = closure()
        eval_count = 0

        # lie to the base algorithm about first closure eval so that if
        # it thinks that the closure has been evaluated N times at the
        # end of its update, then it will be correct.
        # I'm not sure if this is actually important - might be reasonable
        # to just not do this fake closure stuff.
        def skip_once_closure():
            nonlocal eval_count
            eval_count += 1
            if eval_count == 1:
                return loss
            return closure()
    else:
        skip_once_closure = None

    updates = {}
    grads = {}
    deltas = {}

    global_norm = 0.0
    grad_norm = 0.0

    # store gradients and current parameter values.
    # We need to store the gradients because the base optimizer might
    # change them (for example by adding a weight-decay term).
    # We need to store the current parameter values so that we can
    # compute the "update" generated by the base optimizer by subtracting
    # the "new" values from the current values.
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                grads[p] = None
            else:
                grads[p] = p.grad.clone()
            updates[p] = p.data.clone()

    # Re-enable gradients and run the base optimizer step
    torch.set_grad_enabled(prev_grad)
    result = base_step(skip_once_closure)
    torch.set_grad_enabled(False)

    # init state after base_step in case base_step only initializes its
    # own state if self.state is empty.
    # Here, we use the fact that updates[p] is the original value of p before the base step
    _init_state(optimizer, updates, s_decay, betas, s_init, eps, store_delta, log_every)
    mechanic_state = optimizer.state["_mechanic"]

    # compute updates and global norms.
    for group in optimizer.param_groups:
        for p in group["params"]:
            if grads[p] is None:
                continue

            p_ref = mechanic_state[p]["ref"]
            if store_delta:
                deltas[p] = mechanic_state[p]["delta"]
            else:
                # Again, we use updates[p] is the original value of p before base_step
                deltas[p] = (updates[p] - p_ref) / (
                    torch.sum(mechanic_state["s"]) + mechanic_state["eps"]
                )

            updates[p].copy_(p - updates[p])
            p_flat = p.flatten()
            global_norm += torch.dot(p_flat, p_flat)

            g_flat = grads[p].flatten()
            grad_norm += torch.dot(g_flat, g_flat)

    global_norm = torch.sqrt(global_norm)
    grad_norm = torch.sqrt(grad_norm)
    inner_product = 0.0

    # compute inner_product (h in paper pseudocode)
    for group in optimizer.param_groups:
        for p in group["params"]:
            if grads[p] is None:
                continue

            grad = grads[p]

            delta = deltas[p]

            decay = (
                mechanic_state["s_decay"]
                * p.flatten()
                * torch.sum(mechanic_state["s"])
                * grad_norm
                / (global_norm + mechanic_state["eps"])
            )

            inner_product += torch.dot(delta.flatten(), grad.flatten() + decay.flatten())

            delta.add_(updates[p])

    device = inner_product.device

    for key in mechanic_state:
        try:
            if mechanic_state[key].device != device:
                mechanic_state[key] = mechanic_state[key].to(device)
        except AttributeError:
            pass

    # Run the "tuner" step of Mechanic to compute the new s values.
    s = mechanic_state["s"]
    s_decay = mechanic_state["s_decay"]                             # called "lambda" in paper
    s_init = mechanic_state["s_init"]
    betas = mechanic_state["betas"]
    eps = mechanic_state["eps"]
    max_product = mechanic_state["max_product"]                     # called "m" in paper
    reward = mechanic_state["reward"]                               # called "r" in paper
    sum_squared_products = mechanic_state["sum_squared_products"]   # called "v" in paper

    mechanic_state["iter_count"] += 1

    max_product.copy_(torch.maximum(betas * max_product, torch.abs(inner_product)))

    sum_squared_products.mul_(betas**2).add_(torch.square(inner_product))
    reward.mul_(betas).sub_(s * inner_product)
    reward.copy_(torch.clamp(reward, min=torch.zeros_like(reward)))

    wealth = max_product * s_init / len(betas) + reward

    s.copy_(wealth / (torch.sqrt(sum_squared_products) + eps))

    for group in optimizer.param_groups:
        for p in group["params"]:
            if grads[p] is None:
                continue

            p_ref = mechanic_state[p]["ref"]
            delta = deltas[p]
            p.copy_(p_ref + delta * max(torch.sum(s), 0.0))

    log_data = {
        "iter_count": mechanic_state["iter_count"],
        "s": torch.sum(s).item(),
    }

    torch.set_grad_enabled(prev_grad)

    return result, log_data


# Empty class used so that we can do isinstance(mechanize(SGD), Mechanic)
class Mechanic:
    r"""Implements Mechanic, a black-box learning rate tuner for any base optimizer.

    Mechanic tracks the cumulative update direction :math:`\Delta_t` of the base
    optimizer and rescales the total displacement from the reference point
    :math:`\theta_{ref}` (the initial parameters) by a learned scale. With base
    optimizer update :math:`u_t`, gradient :math:`g_t`, and a vector of decay
    rates :math:`\beta \in [0, 1]^n` (all tuner operations are coordinate-wise
    over the :math:`n` decay rates):

    .. math::
       \Delta_{t+1} &= \Delta_t + u_t \\
       h_t &= \Bigl\langle \Delta_t,\, g_t + \lambda \Bigl(\sum_{i=1}^{n} s_{t,i}\Bigr)
           \frac{\lVert g_t \rVert}{\lVert \theta_t \rVert}\, \theta_t \Bigr\rangle \\
       m_t &= \max(\beta m_{t-1},\, \lvert h_t \rvert) \\
       v_t &= \beta^{2} v_{t-1} + h_t^{2} \\
       r_t &= \max(0,\, \beta r_{t-1} - s_t h_t) \\
       W_t &= \frac{s_{init}\, m_t}{n} + r_t \\
       s_{t+1} &= \frac{W_t}{\sqrt{v_t} + \epsilon} \\
       \theta_{t+1} &= \theta_{ref} + \Bigl(\sum_{i=1}^{n} s_{t+1,i}\Bigr) \Delta_{t+1}

    where :math:`\lambda` is the decay factor ``s_decay`` and :math:`s_{init}`
    seeds the initial wealth :math:`W_t`.

    Reference: Ashok Cutkosky, Aaron Defazio, Harsh Mehta,
    "Mechanic: A Learning Rate Tuner", NeurIPS 2023.
    https://arxiv.org/abs/2306.00144

    Note: do not instantiate this class directly. It is a marker base class:
    call ``mechanize(Base)`` to build a tuned subclass of any base optimizer
    class, e.g. ``mechanize(torch.optim.SGD)(model.parameters(), lr=0.01)``.
    """


def is_mechanized(opt):
    return isinstance(opt, Mechanic)


def mechanize(
    Base: Any,
    s_decay: float = 0.01,
    betas: Tuple[float] = (0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999),
    s_init: float = 1e-8,
    eps: float = 1e-8,
    store_delta: bool = False,
    log_func: Any = None,
    log_every: int = 0,
):
    """Wrap a base optimizer class in a mechanic tuner.

    The mechanized optimizer is a subclass of the base optimizer class in
    order to minimize disruption to subsequent code.

    Args:
        Base: base optimizer class to convert into a mechanic instance (e.g. torch.optim.SGD)
        s_decay: how much "weight decay" analog to add (called lambda in the paper).
        betas: list of beta values.
        s_init: initial scale value.
        eps: small number for numerical precision.
        store_delta: whether to store the offsets or recompute them on-the-fly.
        log_func: function to call to log data.
            The input to this function will be a dictionary {'iter_count': iteration count, 's': s_value}
            If None, log_func will be set to:
            def log_func(data):
                logger = logging.getLogger(__name__)
                return logger.info(f"(iter={data['iter_count']}), s_sum (global scaling): {data['s']}")
        log_every: how often (in steps) to call log_func.

    Returns: a new class Mechanized that tunes the base class.

    For example, instead of

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    you could do:

    optimizer = mechanize(torch.optim.SGD)(model.parameters(), lr=0.01)

    The rest of your code should ideally not need to change, even if it accesses
    the internal state of optimizer under the assumption that it is an unadulterated
    instance of torch.optim.SGD.

    Note that this may not always hold: certain libraries like DeepSpeed seem to make
    significant enough assumptions about how the optimizer will work that they may do
    incorrect things.
    """
    if log_func is None:
        logger = logging.getLogger(__name__)

        def log_func(data):
            logger.info(
                f"(iter={data['iter_count']}), s_sum (global scaling): {data['s']}"
            )

    class Mechanized(Base, Mechanic):
        """Wraps a base algorithm as a Mechanic instance."""

        def step(self, closure=None):
            result, log_data = _step(
                self,
                super().step,
                s_decay,
                betas,
                s_init,
                eps,
                store_delta,
                log_every,
                closure,
            )
            mechanic_state = self.state["_mechanic"]
            if log_every > 0 and mechanic_state["iter_count"] % log_every == 0:
                log_func(log_data)

            return result

    Mechanized.__name__ += Base.__name__

    return Mechanized
