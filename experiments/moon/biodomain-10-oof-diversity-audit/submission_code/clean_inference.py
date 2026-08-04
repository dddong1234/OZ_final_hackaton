"""B10 clean inference: full-train fit then one transform/predict pass.

No fixed mutation/cancer constants or external annotations are defined here.
Run only after candidate and train-only ensemble weights are locked.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import subprocess
import sys
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier


CASE = "case_04_shrink10"
COMPONENTS = ("champion", "ovr", "lgbm")
AUTO_PAIR_COUNT = 8
AUTO_GENES_PER_PAIR = 5


def find_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "data" / "raw" / "train.csv").exists():
            return path
    raise FileNotFoundError("Could not find data/raw/train.csv")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clean_context(sdh, train_frame: pd.DataFrame, test_frame: pd.DataFrame, genes: list[str]):
    """Separate parsers; train vocabulary only; no test-wide token summary."""
    safe = sdh._load_safe_submission_module()
    train_cache = safe.RowCache.build(train_frame, genes, show_progress=True)
    test_cache = safe.RowCache.build(
        test_frame, genes, show_progress=True, vocabulary=train_cache.event_names
    )
    train_gene_type, names = sdh._build_gene_type_matrix(
        train_cache.events, len(train_frame), genes
    )
    test_gene_type, _ = sdh._build_gene_type_matrix(
        test_cache.events, len(test_frame), genes, vocabulary=names
    )
    stacked_gene_type = sparse.vstack([train_gene_type, test_gene_type], format="csr")
    return sdh.FeatureContext(safe.RowCache.stack(train_cache, test_cache), stacked_gene_type, names)


def discover_auto_pairs(cache, train_index: np.ndarray, labels: pd.Series, seed: int):
    """Discover pairs from train labels only; no cancer names are hard-coded."""
    y = labels.to_numpy()[train_index]
    matrix = cache.mutation_matrix[train_index]
    classes = np.unique(y)
    predicted = np.empty(len(y), dtype=object)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for fit_rows, holdout_rows in splitter.split(np.zeros(len(y)), y):
        proxy = LogisticRegression(
            solver="lbfgs", C=0.07, max_iter=300,
            class_weight="balanced", random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            proxy.fit(matrix[fit_rows], y[fit_rows])
        predicted[holdout_rows] = proxy.predict(matrix[holdout_rows])

    confusion = confusion_matrix(y, np.asarray(predicted), labels=classes)
    sizes = confusion.sum(axis=1)
    candidates = []
    for left_index in range(len(classes)):
        for right_index in range(left_index + 1, len(classes)):
            swapped = confusion[left_index, right_index] + confusion[right_index, left_index]
            denominator = max(sizes[left_index] + sizes[right_index], 1)
            candidates.append((swapped / denominator, str(classes[left_index]), str(classes[right_index])))
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    return tuple((left, right, AUTO_GENES_PER_PAIR)
                 for _, left, right in candidates[:AUTO_PAIR_COUNT])


def make_models(seed: int):
    lr = dict(solver="lbfgs", C=0.07, max_iter=2000,
              class_weight="balanced", random_state=seed)
    return {
        "champion": LogisticRegression(**lr),
        "ovr": OneVsRestClassifier(LogisticRegression(**lr), n_jobs=1),
        "lgbm": LGBMClassifier(
            objective="multiclass", n_estimators=100, learning_rate=0.05,
            num_leaves=31, class_weight="balanced", random_state=seed,
            n_jobs=-1, deterministic=True, force_col_wise=True, verbosity=-1,
        ),
    }


def load_weights(path: Path) -> dict[int, dict[str, float]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("scope") != "train.csv only; test.csv was not opened or read":
        raise ValueError("Weight report is not certified train-only")
    weights = {}
    for row in report["per_seed"]:
        item = {name: float(value) for name, value in row["three_way_weights"].items()}
        if set(item) != set(COMPONENTS) or not np.isclose(sum(item.values()), 1.0):
            raise ValueError("Invalid three-way weights")
        weights[int(row["seed"])] = item
    return weights


def align(probabilities: np.ndarray, model_classes: np.ndarray, classes: np.ndarray) -> np.ndarray:
    positions = {label: index for index, label in enumerate(model_classes)}
    if set(positions) != set(classes):
        raise ValueError("Component did not learn all 26 classes")
    return probabilities[:, [positions[label] for label in classes]]


def commit_hash(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--weight-report", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--submission-name", default="submission_b10_clean.csv")
    args = parser.parse_args(argv)
    if not args.allow_test:
        raise SystemExit("Refusing to open test.csv without --allow-test")

    root = find_root(Path(__file__).resolve())
    weights_by_seed = load_weights(args.weight_report)
    sdh = load_module(root / "experiments" / "SDH" / "exp_012_enrichment_stability" /
                      "preprocessing.py", "b10_clean_sdh")
    train = pd.read_csv(root / "data" / "raw" / "train.csv")
    test = pd.read_csv(root / "data" / "raw" / "test.csv")
    genes = [column for column in train.columns if column not in ("ID", "SUBCLASS")]
    if list(test.columns) != ["ID", *genes]:
        raise ValueError("Input feature schemas differ")
    if test["ID"].duplicated().any():
        raise ValueError("Duplicate test ID")

    labels = train["SUBCLASS"]
    classes = np.asarray(sorted(labels.unique()))
    context = clean_context(sdh, train[genes], test[genes], genes)
    train_index = np.arange(len(train))
    test_index = np.arange(len(train), len(train) + len(test))

    # Explicitly neutralise every legacy domain-specific field before feature construction.
    original_candidate = sdh.B04_CANDIDATE
    safe_candidate = dataclasses.replace(
        original_candidate,
        experiment_id="b10-clean-auto-train-only",
        backbone="G+B+V+T+R+A+S",
        exact_events=(), gene_pairs=(), gene_groups=(), hotspot_top_k=0,
        contrast_pairs=(), amino_mode="pair", log1p_counts=True,
        b_count_binning=False, lr_max_iter=2000,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    blends, per_seed = [], {}
    started = perf_counter()
    for seed, weights in sorted(weights_by_seed.items()):
        pairs = discover_auto_pairs(context.cache, train_index, labels, seed)
        sdh.B04_CANDIDATE = dataclasses.replace(safe_candidate, contrast_pairs=pairs)
        try:
            train_matrix, test_matrix, _, feature_meta = sdh.build_case_matrices(
                context, train_index, test_index, labels, sdh.make_cases()[CASE], inner_seed=seed
            )
        finally:
            sdh.B04_CANDIDATE = original_candidate

        component_probabilities, warning_counts = {}, {}
        for name, model in make_models(seed).items():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(train_matrix, labels)
            warning_counts[name] = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
            component_probabilities[name] = align(
                model.predict_proba(test_matrix), model.classes_, classes
            )
        blends.append(sum(weights[name] * component_probabilities[name] for name in COMPONENTS))
        per_seed[str(seed)] = {
            "weights": weights,
            "feature_count": int(feature_meta["total_feature_count"]),
            "convergence_warning_count": warning_counts,
            "auto_contrast_pair_count": len(pairs),
        }

    probability = np.mean(np.stack(blends), axis=0)
    submission = pd.DataFrame({"ID": test["ID"], "SUBCLASS": classes[probability.argmax(axis=1)]})
    if list(submission.columns) != ["ID", "SUBCLASS"] or len(submission) != len(test):
        raise AssertionError("Invalid output schema")
    if not submission["ID"].equals(test["ID"]) or submission["ID"].duplicated().any():
        raise AssertionError("ID order or uniqueness check failed")
    if submission["SUBCLASS"].isna().any():
        raise AssertionError("Missing prediction")

    submission_path = args.outdir / args.submission_name
    metadata_path = args.outdir / f"{Path(args.submission_name).stem}_metadata.json"
    submission.to_csv(submission_path, index=False)
    metadata = {
        "experiment_id": "moon-b10-clean-inference",
        "scope": "full train fit then one test transform/predict; no test-derived fit or test-wide summary",
        "git_commit": commit_hash(root),
        "weights_by_seed": weights_by_seed,
        "per_seed": per_seed,
        "output_checks": {"columns": ["ID", "SUBCLASS"], "row_count": int(len(submission)),
                          "duplicate_id_count": 0, "missing_prediction_count": 0},
        "runtime_minutes": (perf_counter() - started) / 60.0,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {submission_path}")
    print(f"Saved {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
