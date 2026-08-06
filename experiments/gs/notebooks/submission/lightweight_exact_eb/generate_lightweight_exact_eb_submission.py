# -*- coding: utf-8 -*-
"""Generate Dacon-format submission CSV from a trained lightweight bundle."""

import argparse
from pathlib import Path

import pandas as pd

from lightweight_exact_eb_core import load_bundle, predict_proba


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    bundle = load_bundle(args.bundle); test = pd.read_csv(args.test_csv); sample = pd.read_csv(args.sample_submission)
    if list(sample.columns) != ["ID", "SUBCLASS"] or not sample.ID.reset_index(drop=True).equals(test.ID.reset_index(drop=True)):
        raise ValueError("sample_submission must have ID/SUBCLASS and preserve test ID order")
    probability = predict_proba(bundle, test)
    output = sample.copy(); output.SUBCLASS = bundle.classes[probability.argmax(axis=1)]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True); output.to_csv(args.output_csv, index=False)
    print({"output": str(args.output_csv), "rows": len(output)})


if __name__ == "__main__":
    main()
