"""Contracts for train-only mutation-profile group construction."""

import importlib.util
from pathlib import Path
import sys

from scipy.sparse import csr_matrix


RUNNER = Path(__file__).with_name("exp_cv_audit_runner.py")
SPEC = importlib.util.spec_from_file_location("exp_cv_audit_runner", RUNNER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_exact_duplicate_profiles_share_a_group() -> None:
    matrix = csr_matrix([[1, 0], [1, 0], [0, 1]], dtype=float)
    groups, diagnostics = MODULE.build_profile_groups(matrix, threshold=0.90)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]
    assert diagnostics["exact_duplicate_rows"] == 2


if __name__ == "__main__":
    test_exact_duplicate_profiles_share_a_group()
    print("CV audit contracts passed")
