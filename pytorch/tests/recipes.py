"""Per-optimizer test recipes.

Optimizers whose defaults work with the generic harness need no entry here.
Add an entry when an optimizer needs a specific learning rate, parameter
shape, gradient type, or stepping protocol.

Modes: "dense" (plain backward + step), "closure" (step(closure)),
"sparse" (sparse gradients), "train_eval" (call opt.train() before stepping
and opt.eval() before measuring, as Schedule-Free requires), "sam" (backward
first so gradients exist, then step(closure), as LookSAM requires).
"""

import torch

DEFAULT = {
    "kwargs": {"lr": 1e-1},
    "steps": 200,
    "shape": (4, 4),
    "mode": "dense",
    "rosenbrock": True,
    "rosenbrock_kwargs": {"lr": 1e-3},
}

RECIPES = {
    "adadelta": {"kwargs": {"lr": 1.0}},
    "adafactor": {"kwargs": {"lr": 1e-1}},
    "asgd": {"kwargs": {"lr": 5e-1}},
    "lbfgs": {"kwargs": {"lr": 1e-1}, "mode": "closure", "rosenbrock": False},
    "muon": {"kwargs": {"lr": 5e-2}, "shape": (8, 8), "rosenbrock": False},
    "sparseadam": {"kwargs": {"lr": 5e-1}, "mode": "sparse", "rosenbrock": False},
    "prodigy": {"kwargs": {"lr": 1.0}, "rosenbrock_kwargs": {"lr": 1.0}},
    "dadaptsgd": {"kwargs": {"lr": 1.0}, "rosenbrock_kwargs": {"lr": 1.0}},
    "dadaptadam": {"kwargs": {"lr": 1.0}, "rosenbrock_kwargs": {"lr": 1.0}},
    "adamg": {"kwargs": {"lr": 1.0}, "rosenbrock_kwargs": {"lr": 1.0}},
    "sgdschedulefree": {"kwargs": {"lr": 5e-1}, "mode": "train_eval", "rosenbrock": False},
    "adamwschedulefree": {"kwargs": {"lr": 5e-1}, "mode": "train_eval", "rosenbrock": False},
    "radamschedulefree": {"kwargs": {"lr": 5e-1}, "mode": "train_eval", "rosenbrock": False},
    "gsam": {"kwargs": {"lr": 1e-1}, "mode": "closure", "rosenbrock": False},
    "wsam": {
        "kwargs": {"base_optimizer": torch.optim.SGD, "lr": 1e-1},
        "mode": "closure",
        "rosenbrock": False,
    },
    "looksam": {
        "kwargs": {"base_optimizer": torch.optim.Adam, "lr": 1e-1, "k": 5},
        "mode": "sam",
        "rosenbrock": False,
    },
    "sam": {
        "kwargs": {"base_optimizer": torch.optim.SGD, "lr": 1e-1},
        "mode": "sam",
        "rosenbrock": False,
    },
    "asam": {
        "kwargs": {"base_optimizer": torch.optim.SGD, "lr": 1e-1},
        "mode": "sam",
        "rosenbrock": False,
    },
    "dog": {"kwargs": {"lr": 1.0}, "rosenbrock_kwargs": {"lr": 1.0}},
    "ldog": {"kwargs": {"lr": 1.0}, "rosenbrock_kwargs": {"lr": 1.0}},
    "momo": {"kwargs": {"lr": 1.0}, "mode": "closure", "rosenbrock": False},
    "momoadam": {"kwargs": {"lr": 1.0}, "mode": "closure", "rosenbrock": False},
    "lion": {"kwargs": {"lr": 1e-2}, "rosenbrock": False},
    "madgrad": {"kwargs": {"lr": 1e-2}, "rosenbrock": False},
    "mirrormadgrad": {"kwargs": {"lr": 1e-2}, "rosenbrock": False},
    "qhm": {
        "kwargs": {"lr": 1e-1, "momentum": 0.9, "nu": 0.7},
        "rosenbrock_kwargs": {"lr": 1e-3, "momentum": 0.9, "nu": 0.7},
    },
    "adan": {"kwargs": {"lr": 1e-1}, "rosenbrock": False},
    "ademamix": {"kwargs": {"lr": 1e-1}, "rosenbrock": False},
    "spam": {"kwargs": {"lr": 1e-1}, "shape": (8, 8), "rosenbrock": False},
    "a2graduni": {"kwargs": {"lr": None, "beta": 10.0, "lips": 10.0}, "rosenbrock": False},
    "a2gradinc": {"kwargs": {"lr": None, "beta": 10.0, "lips": 10.0}, "rosenbrock": False},
    "a2gradexp": {"kwargs": {"lr": None, "beta": 10.0, "lips": 10.0}, "rosenbrock": False},
    "lars": {"kwargs": {"lr": 1e-1}, "rosenbrock": False},
    "tiger": {"kwargs": {"lr": 1e-2}, "rosenbrock": False},
    "ranger": {
        "kwargs": {"lr": 1e-1, "use_gc": False, "degenerated_to_sgd": True},
        "rosenbrock_kwargs": {"lr": 1e-3, "use_gc": False, "degenerated_to_sgd": True},
    },
    "ranger21": {"kwargs": {"lr": 1e-1, "num_iterations": 200}, "rosenbrock": False},
    "shampoo": {"kwargs": {"lr": 5e-1}, "shape": (8, 8), "rosenbrock": False},
    "soap": {"kwargs": {"lr": 5e-2}, "shape": (8, 8), "rosenbrock": False},
    "adamod": {"kwargs": {"lr": 5e-1}, "rosenbrock": False},
    "adamuon": {"kwargs": {"lr": 1e-1}, "shape": (8, 8), "rosenbrock": False},
    "scion": {"kwargs": {"lr": 1e-1}, "shape": (8, 8), "rosenbrock": False},
    "sophiag": {"kwargs": {"lr": 1e-1}, "shape": (4, 4), "rosenbrock": False},
    "splus": {"kwargs": {"lr": 1e-1}, "shape": (8, 8), "rosenbrock": False},
    "normuon": {"kwargs": {"lr": 5e-2}, "shape": (8, 8), "rosenbrock": False},
    "polargrad": {"kwargs": {"lr": 1e-1}, "shape": (8, 8), "rosenbrock": False},
    "focus": {"kwargs": {"lr": 1e-1}, "rosenbrock": False},
    "sgdsai": {"kwargs": {"lr": 1e-1}, "rosenbrock": False},
    "adapnm": {"kwargs": {"lr": 1e-1}, "rosenbrock": False},
    "adagc": {"kwargs": {"lr": 1e-1, "warmup_steps": 10}, "rosenbrock": False},
    "gravity": {"kwargs": {"lr": 2.0}, "rosenbrock": False},
    "signsgd": {"kwargs": {"lr": 1e-2}, "rosenbrock": False},
    "amos": {"kwargs": {"lr": 1e-1}, "rosenbrock": False},
    "adashift": {"kwargs": {"lr": 1e-1}, "rosenbrock": False},
    "came": {"kwargs": {"lr": 5e-2}, "shape": (8, 8), "rosenbrock": False},
    "sm3": {"kwargs": {"lr": 5e-1}, "shape": (8, 8), "rosenbrock": False},
    "grokadamw": {"kwargs": {"lr": 5e-2}, "shape": (8, 8), "rosenbrock": False},
    "galoreadamw": {"kwargs": {"lr": 5e-2}, "shape": (8, 8), "rosenbrock": False},
    "firaadamw": {"kwargs": {"lr": 5e-2}, "shape": (8, 8), "rosenbrock": False},
    "apollo": {"kwargs": {"lr": 5e-2}, "shape": (8, 8), "rosenbrock": False},
    "ldadamw": {
        "kwargs": {"lr": 1e-1, "rank": 2},
        "shape": (8, 8),
        "rosenbrock": False,
    },
}


def recipe_for(name: str) -> dict:
    merged = dict(DEFAULT)
    merged.update(RECIPES.get(name, {}))
    return merged
