# -*- coding: utf-8 -*-
"""Product-facing batch inference: load a saved bundle and emit predictions."""

import argparse
from pathlib import Path

import pandas as pd

from lightweight_exact_eb_core import load_bundle, predict_proba


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    bundle = load_bundle(args.bundle); frame = pd.read_csv(args.input_csv)
    probability = predict_proba(bundle, frame)
    output = pd.DataFrame({"ID": frame.ID, "SUBCLASS": bundle.classes[probability.argmax(axis=1)]})
    for index, label in enumerate(bundle.classes):
        output[f"prob_{label}"] = probability[:, index]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True); output.to_csv(args.output_csv, index=False)
    print({"output": str(args.output_csv), "rows": len(output)})


if __name__ == "__main__":
    main()
