"""Contract for the fixed FM-04 three-seed confirmation candidate."""

import importlib.util
from pathlib import Path
import sys


RUNNER = Path(__file__).with_name("fm_confirm_runner.py")
SPEC = importlib.util.spec_from_file_location("fm_confirm_runner", RUNNER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_fixed_candidate_matches_fm03_winner() -> None:
    candidate = MODULE.fixed_candidate()
    assert candidate.kind == "fm"
    assert candidate.rank == 8
    assert candidate.learning_rate == 3e-4
    assert candidate.class_weight == "balanced"


if __name__ == "__main__":
    test_fixed_candidate_matches_fm03_winner()
    print("FM confirmation contract passed")
