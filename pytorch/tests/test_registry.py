import pytest

import zij


def test_core_optimizers_registered():
    names = zij.list_optimizers()
    for expected in ("adam", "adamw", "sgd", "muon", "adafactor", "lbfgs"):
        assert expected in names


def test_load_optimizer_case_insensitive():
    assert zij.load_optimizer("AdamW") is zij.AdamW
    assert zij.load_optimizer("adamw") is zij.AdamW


def test_load_optimizer_unknown_name():
    with pytest.raises(ValueError, match="unknown optimizer"):
        zij.load_optimizer("does-not-exist")


def test_list_optimizers_wildcard():
    adams = zij.list_optimizers("adam*")
    assert "adam" in adams
    assert "adamw" in adams
    assert "sgd" not in adams


def test_registry_classes_subclass_optimizer():
    # Wrapper optimizers (e.g. Lookahead) compose a base optimizer rather than
    # subclassing Optimizer; everything else must subclass it.
    for name in zij.list_optimizers():
        if zij.construction_of(name) == "wrapper":
            continue
        assert issubclass(zij.load_optimizer(name), zij.Optimizer)
