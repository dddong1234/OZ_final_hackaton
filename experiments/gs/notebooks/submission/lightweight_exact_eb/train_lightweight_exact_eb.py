# -*- coding: utf-8 -*-
"""Train and save the single-LR lightweight Exact-event EB model bundle."""

import argparse
from pathlib import Path

import pandas as pd

from lightweight_exact_eb_core import fit_bundle, save_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--bundle-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train = pd.read_csv(args.train_csv)
    bundle = fit_bundle(train, seed=args.seed)
    save_bundle(bundle, args.bundle_out)
    print(bundle.audit)


if __name__ == "__main__":
    main()
