"""One-command reproduction entry point for the accepted GS seed-bagged submission.

Run this file from any working directory:

    /path/to/.venv/bin/python reproduce_h0_selective_eb_3seed.py

It fits the predeclared seeds (42, 777, 2024) on the full train split, uses
test only for train-fitted transformation/prediction, equally averages the
three probability matrices, and writes a Dacon submission plus an audit JSON.

This is an execution entry point: the implementation it calls is maintained
inside experiments/gs only, and no team-member or external experiment code is
imported at runtime.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


SEEDS = (42, 777, 2024)
OUTPUT_NAME = "submission_h0_selective_eb_lr_lgbm_specialist_seed42_777_2024_bagged.csv"


def project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "data" / "raw" / "train.csv").exists():
            return candidate
    raise FileNotFoundError("Could not locate project data/raw/train.csv")


def main() -> None:
    root = project_root()
    gs_common = root / "experiments" / "gs" / "notebooks" / "exp_model_007" / "common"
    runner = gs_common / "h0_selective_eb_submission.py"
    if not runner.exists():
        raise FileNotFoundError(f"Required GS submission implementation is missing: {runner}")
    if str(gs_common) not in sys.path:
        sys.path.insert(0, str(gs_common))

    # GS-only implementation; it verifies equal weights and submission schema.
    from h0_selective_eb_submission import run_seed_bagged  # noqa: PLC0415

    destination = run_seed_bagged(output_name=OUTPUT_NAME, seeds=SEEDS)
    audit_path = destination.with_suffix(".audit.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = {
        "seeds": list(SEEDS),
        "seed_weights": [1 / 3, 1 / 3, 1 / 3],
        "leakage_check": True,
        "nan_as_mutation_count": 0,
        "raw_train_test_concat": False,
    }
    for key, value in expected.items():
        if audit.get(key) != value:
            raise AssertionError(f"reproduction audit mismatch: {key}={audit.get(key)!r}")
    print(json.dumps({"submission": str(destination), "audit": str(audit_path), **expected}, ensure_ascii=False))


if __name__ == "__main__":
    main()
