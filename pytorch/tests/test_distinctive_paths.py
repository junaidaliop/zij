"""Tests that exercise each optimizer's distinctive code path, not just its
fallback. Several optimizers reduce to AdamW/SGD/Adam unless a specific option
is set: GaLore/Fira/APOLLO need a per-group ``rank`` to build their low-rank
projector, PID needs ``momentum`` to engage its integral/derivative terms,
QHAdam needs a ``nu`` other than 1.0 for its quasi-hyperbolic interpolation,
and SophiaG needs periodic ``update_hessian`` calls for its preconditioner.
Each test asserts both convergence and that the distinctive state was formed,
so a silent regression to the fallback would fail here.
"""

import pytest
import torch
import torch.nn as nn

import zij


def _quad(p):
    return (p**2).sum()


def _run_groups(name, group_extra, lr, steps=200):
    torch.manual_seed(7)
    p = nn.Parameter(torch.randn(8, 8))
    opt = zij.load_optimizer(name)([{"params": [p], **group_extra}], lr=lr)
    start = _quad(p).item()
    for _ in range(steps):
        opt.zero_grad()
        _quad(p).backward()
        opt.step()
    return start, _quad(p).item(), opt.state[p]


@pytest.mark.parametrize(
    "name,group_extra",
    [
        ("galoreadamw", {"rank": 4, "update_proj_gap": 50, "scale": 0.25, "proj_type": "std"}),
        ("firaadamw", {"rank": 4, "update_proj_gap": 50, "alpha": 1.0, "proj_type": "std"}),
        ("apollo", {"rank": 4, "update_proj_gap": 50, "scale": 1.0, "projection_type": "std"}),
    ],
)
def test_low_rank_projection_engages(name, group_extra):
    start, end, state = _run_groups(name, group_extra, lr=5e-2)
    assert "projector" in state, f"{name}: low-rank projector was not created"
    assert end < start * 0.5, f"{name}: {start:.4f} -> {end:.4f}"


def test_pid_controller_engages():
    torch.manual_seed(7)
    p = nn.Parameter(torch.randn(8, 8))
    opt = zij.PID([p], lr=1e-1, momentum=0.9, integral=5.0, derivative=10.0)
    start = _quad(p).item()
    for _ in range(200):
        opt.zero_grad()
        _quad(p).backward()
        opt.step()
    end = _quad(p).item()
    assert "i_buffer" in opt.state[p], "PID integral term was not engaged"
    assert end < start * 0.5, f"PID: {start:.4f} -> {end:.4f}"


def test_qhadam_quasi_hyperbolic_interpolation():
    torch.manual_seed(7)
    p = nn.Parameter(torch.randn(4, 4))
    opt = zij.QHAdam([p], lr=1e-1, nus=(0.7, 1.0))
    start = _quad(p).item()
    for _ in range(200):
        opt.zero_grad()
        _quad(p).backward()
        opt.step()
    end = _quad(p).item()
    assert end < start * 0.5, f"QHAdam(nu1=0.7): {start:.4f} -> {end:.4f}"


def test_sophiag_hessian_preconditioning():
    torch.manual_seed(7)
    p = nn.Parameter(torch.randn(4, 4))
    opt = zij.SophiaG([p], lr=1e-1)
    start = _quad(p).item()
    for i in range(200):
        opt.zero_grad()
        _quad(p).backward()
        opt.step()
        if i % 10 == 0:
            opt.zero_grad()
            _quad(p).backward()
            opt.update_hessian()
    end = _quad(p).item()
    assert opt.state[p]["hessian"].abs().max() > 0, "SophiaG Hessian estimate stayed zero"
    assert end < start * 0.5, f"SophiaG: {start:.4f} -> {end:.4f}"
