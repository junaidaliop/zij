"""Convergence tests for optimizers that take a model rather than a parameter list.

AdamMini reads parameter names to group the embedding and head; Lomo and
AdaLomo fuse the gradient step into the backward pass. These do not fit the
generic parameter-list harness, so they are exercised on a small model here.
"""

import pytest
import torch
import torch.nn as nn

import zij


def _make_problem():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 1))
    inputs = torch.randn(32, 8)
    targets = torch.randn(32, 1)
    loss_fn = nn.MSELoss()
    return model, inputs, targets, loss_fn


def test_adam_mini_converges():
    model, x, y, loss_fn = _make_problem()
    opt = zij.AdamMini(model, lr=5e-2, weight_decay=0.0)
    start = loss_fn(model(x), y).item()
    for _ in range(150):
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    end = loss_fn(model(x), y).item()
    assert end < start * 0.7, f"{start:.4f} -> {end:.4f}"


@pytest.mark.parametrize("name", ["Lomo", "AdaLomo"])
def test_lomo_fused_backward_converges(name):
    model, x, y, loss_fn = _make_problem()
    opt = getattr(zij, name)(model, lr=5e-2)
    start = loss_fn(model(x), y).item()
    for _ in range(150):
        loss = loss_fn(model(x), y)
        opt.fused_backward(loss, 5e-2)
    end = loss_fn(model(x), y).item()
    assert end < start * 0.7, f"{name}: {start:.4f} -> {end:.4f}"


def test_adahessian_converges():
    # Adahessian needs gradients built with create_graph=True for its
    # Hutchinson Hessian-trace estimate.
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = zij.Adahessian([p], lr=1e-1)
    start = (p**2).sum().item()
    for _ in range(150):
        opt.zero_grad()
        (p**2).sum().backward(create_graph=True)
        opt.step()
    end = (p**2).sum().item()
    assert end < start * 0.5, f"Adahessian: {start:.4f} -> {end:.4f}"


def test_lookahead_wraps_base_optimizer():
    import torch

    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = zij.Lookahead(zij.AdamW([p], lr=1e-1), k=5, alpha=0.5)
    start = (p**2).sum().item()
    for _ in range(200):
        opt.zero_grad()
        (p**2).sum().backward()
        opt.step()
    end = (p**2).sum().item()
    assert end < start * 0.5, f"Lookahead: {start:.4f} -> {end:.4f}"


def test_trac_wraps_base_optimizer():
    import torch

    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = zij.TRAC(zij.AdamW([p], lr=1e-1))
    start = (p**2).sum().item()
    for _ in range(200):
        opt.zero_grad()
        (p**2).sum().backward()
        opt.step()
    end = (p**2).sum().item()
    assert end < start * 0.5, f"TRAC: {start:.4f} -> {end:.4f}"


def test_schedulefree_wrapper_wraps_base_optimizer():
    import torch

    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = zij.ScheduleFreeWrapper(zij.AdamW([p], lr=1e-1), momentum=0.9)
    opt.train()
    start = (p**2).sum().item()
    for _ in range(200):
        opt.zero_grad()
        (p**2).sum().backward()
        opt.step()
    opt.eval()
    end = (p**2).sum().item()
    assert end < start * 0.5, f"ScheduleFreeWrapper: {start:.4f} -> {end:.4f}"


def test_block_optimizer_cycles_blocks():
    # BlockOptimizer (BAdam) trains one named block at a time, switching
    # periodically; it needs named parameters, so it runs on a small model and
    # trains long enough to cycle through every block.
    model, x, y, loss_fn = _make_problem()
    opt = zij.BlockOptimizer(
        model.named_parameters(),
        base_optimizer=zij.AdamW,
        block_prefix_list=[["0"], ["2"]],
        switch_block_every=20,
        switch_mode="descending",
        verbose=0,
    )
    start = loss_fn(model(x), y).item()
    for _ in range(300):
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    end = loss_fn(model(x), y).item()
    assert end < start * 0.7, f"BlockOptimizer: {start:.4f} -> {end:.4f}"
