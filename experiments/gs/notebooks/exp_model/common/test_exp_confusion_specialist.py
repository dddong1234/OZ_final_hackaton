"""Contracts for fold-local confusion-pair selection and conservative gating."""

import importlib.util
from pathlib import Path
import sys

import numpy as np


RUNNER = Path(__file__).with_name("exp_confusion_specialist_runner.py")
SPEC = importlib.util.spec_from_file_location("exp_confusion_specialist_runner", RUNNER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_select_disjoint_pairs_from_inner_oof_confusions() -> None:
    classes = ["A", "B", "C", "D"]
    truth = np.array(["A", "A", "A", "B", "B", "B", "C", "C", "D", "D"])
    prediction = np.array(["B", "B", "C", "A", "A", "C", "D", "D", "C", "C"])

    pairs = MODULE.select_disjoint_confusion_pairs(truth, prediction, classes, max_pairs=2)

    assert [(item.left, item.right, item.count) for item in pairs] == [("A", "B", 4), ("C", "D", 4)]


def test_soft_gate_preserves_pair_mass_and_other_classes() -> None:
    primary = np.array([[0.10, 0.40, 0.30, 0.20], [0.05, 0.05, 0.50, 0.40]], dtype=np.float32)
    specialist = np.array([[0.80, 0.20], [0.90, 0.10]], dtype=np.float32)

    corrected, applied = MODULE.apply_pair_gate(primary, specialist, left_index=0, right_index=1, alpha=0.30, min_pair_mass=0.20)

    assert applied.tolist() == [True, False]
    np.testing.assert_allclose(corrected[0, :2].sum(), primary[0, :2].sum())
    np.testing.assert_allclose(corrected[0, 2:], primary[0, 2:])
    np.testing.assert_allclose(corrected[1], primary[1])
    np.testing.assert_allclose(corrected.sum(axis=1), np.ones(2))


if __name__ == "__main__":
    test_select_disjoint_pairs_from_inner_oof_confusions()
    test_soft_gate_preserves_pair_mass_and_other_classes()
    print("Confusion-specialist contracts passed")
