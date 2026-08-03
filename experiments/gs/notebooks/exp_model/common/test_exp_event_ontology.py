"""Contracts for event ontology parsing and fixed position bins."""

import importlib.util
from pathlib import Path
import sys


RUNNER = Path(__file__).with_name("exp_event_ontology_runner.py")
SPEC = importlib.util.spec_from_file_location("exp_event_ontology_runner", RUNNER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_position_and_fixed_bin_are_parsed_without_labels() -> None:
    assert MODULE.event_position("R132H") == 132
    assert MODULE.event_position("E545K") == 545
    assert MODULE.position_bin(132) == "101_150"
    assert MODULE.event_position("FS") is None


if __name__ == "__main__":
    test_position_and_fixed_bin_are_parsed_without_labels()
    print("Event ontology contracts passed")
