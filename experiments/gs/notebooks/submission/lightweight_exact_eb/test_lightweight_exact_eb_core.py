# -*- coding: utf-8 -*-
"""Run directly with Python when pytest is unavailable."""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from lightweight_exact_eb_core import fit_bundle, load_bundle, normalise_cell, predict_proba, save_bundle


HERE = Path(__file__).resolve().parent


def toy_train() -> pd.DataFrame:
    rows = []
    for index in range(10):
        rows.append({"ID": f"a{index}", "G1": "R1H", "G2": "WT", "SUBCLASS": "A"})
        rows.append({"ID": f"b{index}", "G1": "WT", "G2": "V2E", "SUBCLASS": "B"})
    return pd.DataFrame(rows)


def test_missing_values_never_create_events() -> None:
    assert normalise_cell(np.nan) == ()
    assert normalise_cell("") == ()
    assert normalise_cell("WT") == ()
    assert normalise_cell("p.R1H R1H") == ("R1H",)


def test_apply_ignores_test_only_exact_event() -> None:
    train = toy_train(); bundle = fit_bundle(train, seed=42)
    frame = pd.DataFrame({"ID": ["x"], "G1": ["A99V"], "G2": ["WT"]})
    probability = predict_proba(bundle, frame)
    assert probability.shape == (1, 2)
    assert np.allclose(probability.sum(axis=1), 1.0)


def test_bundle_round_trip_and_cli_submission() -> None:
    train = toy_train()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir); train_csv = temp / "train.csv"; test_csv = temp / "test.csv"; sample_csv = temp / "sample_submission.csv"; bundle_path = temp / "model.joblib"; output = temp / "submission.csv"
        train.to_csv(train_csv, index=False); test = train.drop(columns="SUBCLASS").iloc[:4].copy(); test.to_csv(test_csv, index=False); pd.DataFrame({"ID": test.ID, "SUBCLASS": ""}).to_csv(sample_csv, index=False)
        bundle = fit_bundle(train, seed=42); save_bundle(bundle, bundle_path)
        assert np.allclose(predict_proba(bundle, test), predict_proba(load_bundle(bundle_path), test))
        command = [sys.executable, str(HERE / "generate_lightweight_exact_eb_submission.py"), "--bundle", str(bundle_path), "--test-csv", str(test_csv), "--sample-submission", str(sample_csv), "--output-csv", str(output)]
        assert subprocess.run(command, check=False, capture_output=True, text=True).returncode == 0
        result = pd.read_csv(output); assert list(result.columns) == ["ID", "SUBCLASS"]; assert result.ID.tolist() == test.ID.tolist()


def test_cv_cli_is_train_only() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir); train_csv = temp / "train.csv"; result_dir = temp / "cv"
        toy_train().to_csv(train_csv, index=False)
        command = [sys.executable, str(HERE / "evaluate_lightweight_exact_eb_cv.py"), "--train-csv", str(train_csv), "--result-dir", str(result_dir)]
        assert subprocess.run(command, check=False, capture_output=True, text=True).returncode == 0
        summary = pd.read_csv(result_dir / "lightweight_exact_eb_3seed_summary.csv")
        audit = (result_dir / "lightweight_exact_eb_leakage_audit.json").read_text(encoding="utf-8")
        assert len(summary) == 3 and summary.leakage_check.all() and summary.nan_as_mutation_count.eq(0).all()
        assert '"test_read": false' in audit


if __name__ == "__main__":
    test_missing_values_never_create_events(); test_apply_ignores_test_only_exact_event(); test_bundle_round_trip_and_cli_submission(); test_cv_cli_is_train_only()
    print("lightweight Exact-event EB tests passed")
