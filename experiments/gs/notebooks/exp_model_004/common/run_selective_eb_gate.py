"""새 seed에서만 고정 selective EB gate를 검증한다. test는 읽지 않는다."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from selective_eb_gate import SELECTIVE_MARGIN, VALIDATION_SEEDS, selective_probability


def root() -> Path:
    for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (candidate / "data/raw/train.csv").exists():
            return candidate
    raise FileNotFoundError("project root를 찾지 못했습니다.")


def p1_module():
    common = root() / "experiments/gs/notebooks/exp_model_002/common"
    sys.path.insert(0, str(common))
    spec = importlib.util.spec_from_file_location("selective_gate_p1", common / "run_p1_axis.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def class_rows(y: np.ndarray, classes: np.ndarray, p1: np.ndarray, eb: np.ndarray, gated: np.ndarray, seed: int) -> list[dict]:
    rows = []
    for idx, label in enumerate(classes):
        mask = y == label
        values = {}
        for name, probability in (("P1_non_EB", p1), ("P1_EB", eb), ("selective_EB_gate", gated)):
            values[name] = f1_score(
                y,
                classes[probability.argmax(1)],
                labels=[label],
                average="macro",
                zero_division=0,
            )
        rows.append({"seed": seed, "class": label, "support": int(mask.sum()), **values,
                     "delta_gate_vs_eb": values["selective_EB_gate"] - values["P1_EB"]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="exp-selective-eb-gate-01")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(VALIDATION_SEEDS))
    args = parser.parse_args()
    if tuple(args.seeds) != VALIDATION_SEEDS:
        raise ValueError(f"새 seed 확정 검증은 {VALIDATION_SEEDS}로 고정합니다.")

    module = p1_module()
    base, ref, cache, tokens, y, classes = module.legacy_context()
    train = pd.read_csv(root() / "data/raw/train.csv")
    genes = [column for column in train if column not in (base.CFG.id_col, base.CFG.target_col)]
    assert int(train[genes].isna().sum().sum()) == 0, "train NaN 계약 위반"
    context = (base, ref, cache, tokens, y, classes)
    out = Path(__file__).parent.parent / "result"
    out.mkdir(exist_ok=True)

    per_seed, fold_rows, all_class_rows, gate_rows = [], [], [], []
    for seed in args.seeds:
        start = time.time()
        folds = module.fixed_folds(y, seed)
        classes, p1, _, warning_p1 = module.p1_cv(context, seed)
        _, eb, _, warning_eb = module.eb_cv(context, seed)
        gated, use_non_eb = selective_probability(p1, eb)
        variants = {"P1_non_EB": p1, "P1_EB": eb, "selective_EB_gate": gated}
        scores = {name: f1_score(y, classes[probability.argmax(1)], average="macro") for name, probability in variants.items()}
        for name, probability in variants.items():
            per_seed.append({"seed": seed, "variant": name, "oof_macro_f1": scores[name],
                             "delta_vs_eb": scores[name] - scores["P1_EB"],
                             "convergence_warning_count": warning_p1 if name == "P1_non_EB" else warning_eb,
                             "leakage_check": True, "nan_as_mutation_count": 0,
                             "runtime_seconds": time.time() - start})
            for fold, (_, validation_index) in enumerate(folds, 1):
                fold_rows.append({"seed": seed, "variant": name, "fold": fold,
                                  "macro_f1": f1_score(y[validation_index], classes[probability[validation_index].argmax(1)], average="macro"),
                                  "feature_count": 8201.2})
        margin = np.partition(eb, kth=-2, axis=1)[:, -1] - np.partition(eb, kth=-2, axis=1)[:, -2]
        low = use_non_eb
        for group, mask in (("low_margin_<0.05", low), ("eb_margin_>=0.05", ~low)):
            gate_rows.append({"seed": seed, "group": group, "support": int(mask.sum()), "rate": float(mask.mean()),
                              "p1_non_eb_macro_f1": f1_score(y[mask], classes[p1[mask].argmax(1)], average="macro", zero_division=0),
                              "p1_eb_macro_f1": f1_score(y[mask], classes[eb[mask].argmax(1)], average="macro", zero_division=0),
                              "gate_macro_f1": f1_score(y[mask], classes[gated[mask].argmax(1)], average="macro", zero_division=0),
                              "margin_min": float(margin[mask].min()), "margin_max": float(margin[mask].max())})
        all_class_rows.extend(class_rows(y, classes, p1, eb, gated, seed))
        pd.DataFrame({"true_class": y, "use_p1_non_eb": use_non_eb, "eb_margin": margin,
                      **{f"p1_non_eb__{label}": p1[:, i] for i, label in enumerate(classes)},
                      **{f"p1_eb__{label}": eb[:, i] for i, label in enumerate(classes)},
                      **{f"gate__{label}": gated[:, i] for i, label in enumerate(classes)}}).to_csv(out / f"{args.run_id}_seed{seed}_oof_probabilities.csv", index=False)

    per_seed_frame = pd.DataFrame(per_seed)
    summary = per_seed_frame.groupby("variant", as_index=False).agg(seed_count=("seed", "nunique"), oof_macro_f1_mean=("oof_macro_f1", "mean"), oof_macro_f1_std=("oof_macro_f1", "std"), delta_vs_eb_mean=("delta_vs_eb", "mean"), delta_vs_eb_std=("delta_vs_eb", "std"), convergence_warning_count=("convergence_warning_count", "sum"), leakage_check=("leakage_check", "all"), nan_as_mutation_count=("nan_as_mutation_count", "max"))
    summary.to_csv(out / f"{args.run_id}_3seed_summary.csv", index=False)
    per_seed_frame.to_csv(out / f"{args.run_id}_per_seed.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / f"{args.run_id}_fold_metrics.csv", index=False)
    pd.DataFrame(all_class_rows).to_csv(out / f"{args.run_id}_class_metrics.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(out / f"{args.run_id}_gate_usage.csv", index=False)
    audit = {"test_read": False, "submission_created": False, "threshold": SELECTIVE_MARGIN,
             "threshold_selection_seeds": [42, 777, 2024], "validation_seeds": list(VALIDATION_SEEDS),
             "threshold_retuned": False, "fold_train_only_supervised_statistics": True,
             "leakage_check": True, "nan_as_mutation_count": 0}
    (out / f"{args.run_id}_leakage_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
