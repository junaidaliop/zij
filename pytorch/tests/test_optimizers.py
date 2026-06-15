import pytest
import torch
from recipes import recipe_for

import zij

# Optimizers built from a parameter iterable run in this generic harness;
# model-based and wrapper optimizers are covered by test_model_based.py.
ALL_NAMES = [n for n in zij.list_optimizers() if zij.construction_of(n) == "params"]


def quadratic_loss(p, target):
    return ((p - target) ** 2).sum()


def run_quadratic(name):
    recipe = recipe_for(name)
    torch.manual_seed(7)
    p = torch.nn.Parameter(torch.randn(*recipe["shape"]))
    target = torch.zeros_like(p)
    opt = zij.load_optimizer(name)([p], **recipe["kwargs"])
    start = quadratic_loss(p, target).item()
    mode = recipe["mode"]

    if mode == "train_eval":
        opt.train()
    for _ in range(recipe["steps"]):
        if mode == "closure":
            def closure():
                opt.zero_grad()
                loss = quadratic_loss(p, target)
                loss.backward()
                return loss
            opt.step(closure)
        elif mode == "sam":
            def closure():
                opt.zero_grad()
                loss = quadratic_loss(p, target)
                loss.backward()
                return loss
            closure()
            opt.step(closure)
        elif mode == "sparse":
            opt.zero_grad()
            p.grad = (2 * (p.detach() - target)).to_sparse()
            opt.step()
        else:
            opt.zero_grad()
            quadratic_loss(p, target).backward()
            opt.step()
    if mode == "train_eval":
        opt.eval()

    end = quadratic_loss(p, target).item()
    return start, end, opt


@pytest.mark.parametrize("name", ALL_NAMES)
def test_quadratic_convergence(name):
    start, end, _ = run_quadratic(name)
    assert torch.isfinite(torch.tensor(end)), f"{name} diverged to non-finite loss"
    assert end < start * 0.5, f"{name}: {start:.4f} -> {end:.4f}"


def _deep_equal(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and torch.equal(a, b)
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_deep_equal(x, y) for x, y in zip(a, b))
    return a == b


@pytest.mark.parametrize("name", ALL_NAMES)
def test_state_dict_roundtrip(name):
    _, _, opt = run_quadratic(name)
    state = opt.state_dict()
    opt.load_state_dict(state)
    assert _deep_equal(opt.state_dict()["param_groups"], state["param_groups"])


@pytest.mark.parametrize(
    "name", [n for n in ALL_NAMES if recipe_for(n)["rosenbrock"]]
)
def test_rosenbrock_descent(name):
    recipe = recipe_for(name)
    torch.manual_seed(7)
    p = torch.nn.Parameter(torch.tensor([-1.5, 2.0]))
    opt = zij.load_optimizer(name)([p], **recipe["rosenbrock_kwargs"])

    def rosenbrock(t):
        x, y = t[0], t[1]
        return (1 - x) ** 2 + 100 * (y - x**2) ** 2

    start = rosenbrock(p).item()
    for _ in range(300):
        opt.zero_grad()
        rosenbrock(p).backward()
        opt.step()
    end = rosenbrock(p).item()

    assert torch.isfinite(torch.tensor(end)), f"{name} diverged on Rosenbrock"
    assert end < start, f"{name}: {start:.4f} -> {end:.4f}"
