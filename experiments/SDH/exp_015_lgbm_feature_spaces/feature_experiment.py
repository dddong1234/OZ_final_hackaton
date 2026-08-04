"""SDH exp_015: fixed-LGBM feature-space ablations.

Every learned representation is fitted inside the outer-fold training split.
Validation rows are only transformed/applied.  No train/test concatenation is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier


ROOT = Path(__file__).resolve().parents[3]
PIPELINE_PATH = ROOT / "experiments/SDH/exp_013_standalone_pipeline_audit/standalone_pipeline.py"


def _load_pipeline():
    spec = importlib.util.spec_from_file_location("sdh_exp13_pipeline", PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {PIPELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


P13 = _load_pipeline()


@dataclass(frozen=True)
class FeatureCase:
    name: str
    include_prefixes: tuple[str, ...] | None = None
    include_names: tuple[str, ...] = ()
    gene_min_support: int | None = None
    add_fixed_bins: bool = False
    gain_top_k: int | None = None
    description: str = ""


@dataclass
class PreparedFold:
    fold: int
    fit_index: np.ndarray
    valid_index: np.ndarray
    y_fit: np.ndarray
    y_valid: np.ndarray
    x_fit: sparse.csr_matrix
    x_valid: sparse.csr_matrix
    names: list[str]


CORE_PREFIXES = ("B__", "V__", "S__", "E__")
AGGREGATE_NAMES = (
    "T__truncating_gene_count",
    "R__recurrent_missense_event_count",
)


CASES: tuple[FeatureCase, ...] = (
    FeatureCase("F00_full", description="exp13 전체 피처 기준선"),
    FeatureCase("F01_core", CORE_PREFIXES, AGGREGATE_NAMES, description="집계형 core만"),
    FeatureCase("F02_core_no_E", ("B__", "V__", "S__"), AGGREGATE_NAMES, description="core에서 supervised enrichment 제거"),
    FeatureCase("F03_E_only", ("E__",), description="class-enrichment score만"),
    FeatureCase("F04_E_BVS", ("E__", "B__", "V__", "S__"), description="enrichment + row-local 집계"),
    FeatureCase("F05_G_E", ("G__", "E__"), description="mutation gene + enrichment"),
    FeatureCase("F06_core_G", CORE_PREFIXES + ("G__",), AGGREGATE_NAMES, description="core + 모든 mutation gene"),
    FeatureCase("F07_core_G_sup2", CORE_PREFIXES + ("G__",), AGGREGATE_NAMES, 2, description="core + support>=2 gene"),
    FeatureCase("F08_core_G_sup5", CORE_PREFIXES + ("G__",), AGGREGATE_NAMES, 5, description="core + support>=5 gene"),
    FeatureCase("F09_core_G_sup10", CORE_PREFIXES + ("G__",), AGGREGATE_NAMES, 10, description="core + support>=10 gene"),
    FeatureCase("F10_core_G_sup20", CORE_PREFIXES + ("G__",), AGGREGATE_NAMES, 20, description="core + support>=20 gene"),
    FeatureCase("F11_core_G_sup30", CORE_PREFIXES + ("G__",), AGGREGATE_NAMES, 30, description="core + support>=30 gene"),
    FeatureCase("F12_core_G_sup50", CORE_PREFIXES + ("G__",), AGGREGATE_NAMES, 50, description="core + support>=50 gene"),
    FeatureCase("F13_core_T", CORE_PREFIXES + ("T__",), AGGREGATE_NAMES, description="core + gene별 truncation"),
    FeatureCase("F14_core_R", CORE_PREFIXES + ("R__",), AGGREGATE_NAMES, description="core + recurrent missense"),
    FeatureCase("F15_core_A", CORE_PREFIXES + ("A_pair__",), AGGREGATE_NAMES, description="core + amino-acid substitution pair"),
    FeatureCase("F16_core_R_A", CORE_PREFIXES + ("R__", "A_pair__"), AGGREGATE_NAMES, description="core + recurrent missense + amino pair"),
    FeatureCase("F17_no_G", ("B__", "V__", "T__", "R__", "A_pair__", "S__", "E__"), description="full에서 mutation gene 제거"),
    FeatureCase("F18_no_A", ("G__", "B__", "V__", "T__", "R__", "S__", "E__"), description="full에서 amino pair 제거"),
    FeatureCase("F19_no_T_R", ("G__", "B__", "V__", "A_pair__", "S__", "E__"), description="full에서 truncation/recurrent 제거"),
    FeatureCase("F20_full_bins", add_fixed_bins=True, description="full + 고정 count bins"),
    FeatureCase("F21_core_G10_bins", CORE_PREFIXES + ("G__",), AGGREGATE_NAMES, 10, True, description="support>=10 + bins"),
    FeatureCase("F22_full_top250", gain_top_k=250, description="fold-train gain Top-250"),
    FeatureCase("F23_full_top500", gain_top_k=500, description="fold-train gain Top-500"),
    FeatureCase("F24_full_top1000", gain_top_k=1000, description="fold-train gain Top-1000"),
    FeatureCase("F25_full_top2000", gain_top_k=2000, description="fold-train gain Top-2000"),
)
CASE_MAP = {case.name: case for case in CASES}


GROUPS = {
    "representation": [f"F{i:02d}_" for i in range(0, 7)],
    "gene_support": [f"F{i:02d}_" for i in range(7, 13)],
    "blocks_and_bins": [f"F{i:02d}_" for i in range(13, 22)],
    "gain_topk": [f"F{i:02d}_" for i in range(22, 26)],
}


def cases_for_group(group: str) -> list[FeatureCase]:
    prefixes = GROUPS[group]
    return [case for case in CASES if any(case.name.startswith(p) for p in prefixes)]


def make_lgbm(seed: int, *, selector: bool = False) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        boosting_type="gbdt",
        n_estimators=100 if selector else 400,
        learning_rate=0.05,
        num_leaves=15 if selector else 25,
        reg_alpha=0.0,
        reg_lambda=0.0,
        min_child_samples=10,
        min_child_weight=1e-3,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def prepare_seed(train: pd.DataFrame, genes: list[str], seed: int = 42) -> list[PreparedFold]:
    labels = train["SUBCLASS"].reset_index(drop=True)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds: list[PreparedFold] = []
    for fold, (fit_index, valid_index) in enumerate(splitter.split(train, labels), 1):
        y_fit = labels.iloc[fit_index].to_numpy()
        y_valid = labels.iloc[valid_index].to_numpy()
        x_fit, x_valid, names, audit = P13.build_design_matrices(
            train.iloc[fit_index][genes].reset_index(drop=True),
            train.iloc[valid_index][genes].reset_index(drop=True),
            y_fit,
            genes,
            seed=seed * 100 + fold,
            use_fixed_contrast=False,
        )
        safe_columns = np.array(
            [not name.startswith(("C__", "D__exact_")) for name in names]
        )
        x_fit, x_valid = x_fit[:, safe_columns], x_valid[:, safe_columns]
        names = [name for name, keep in zip(names, safe_columns) if keep]
        assert not any(name.startswith(("C__", "D__exact_")) for name in names)
        assert audit["raw_train_test_concat"] is False
        assert audit["vocabulary_source"] == "fit_frame_only"
        folds.append(PreparedFold(fold, fit_index, valid_index, y_fit, y_valid, x_fit, x_valid, names))
        print(f"prepared fold={fold} features={len(names):,}")
    return folds


def _column_mask(fold: PreparedFold, case: FeatureCase) -> np.ndarray:
    if case.include_prefixes is None:
        mask = np.ones(len(fold.names), dtype=bool)
    else:
        mask = np.array([
            name in case.include_names or name.startswith(case.include_prefixes)
            for name in fold.names
        ])
    if case.gene_min_support is not None:
        support = np.asarray(fold.x_fit.getnnz(axis=0)).ravel()
        for index, name in enumerate(fold.names):
            if name.startswith("G__") and support[index] < case.gene_min_support:
                mask[index] = False
    if not mask.any():
        raise ValueError(f"{case.name} selected zero columns")
    return mask


def _fixed_bins(matrix: sparse.csr_matrix, names: list[str]) -> sparse.csr_matrix:
    # Only fixed, row-wise count columns are binned. No dataset statistic is fitted.
    eligible = [
        i for i, name in enumerate(names)
        if name.startswith(("B__", "V__")) or name in AGGREGATE_NAMES
    ]
    if not eligible:
        return sparse.csr_matrix((matrix.shape[0], 0), dtype=np.float32)
    values = matrix[:, eligible].toarray()
    # B/V were log1p-transformed by exp13; aggregate T/R remained raw counts.
    for j, index in enumerate(eligible):
        if names[index].startswith(("B__", "V__")):
            values[:, j] = np.expm1(values[:, j])
    boundaries = ((0, 0), (1, 1), (2, 2), (3, 4), (5, 7), (8, np.inf))
    columns = [((values >= low) & (values <= high)).astype(np.float32) for low, high in boundaries]
    return sparse.csr_matrix(np.hstack(columns))


def matrices_for_case(fold: PreparedFold, case: FeatureCase, seed: int) -> tuple[sparse.csr_matrix, sparse.csr_matrix, int]:
    mask = _column_mask(fold, case)
    x_fit = fold.x_fit[:, mask]
    x_valid = fold.x_valid[:, mask]
    selected_names = [name for name, keep in zip(fold.names, mask) if keep]
    if case.add_fixed_bins:
        x_fit = sparse.hstack([x_fit, _fixed_bins(fold.x_fit, fold.names)], format="csr")
        x_valid = sparse.hstack([x_valid, _fixed_bins(fold.x_valid, fold.names)], format="csr")
    if case.gain_top_k is not None and x_fit.shape[1] > case.gain_top_k:
        selector = make_lgbm(seed, selector=True)
        selector.fit(x_fit, fold.y_fit)
        gain = selector.booster_.feature_importance(importance_type="gain")
        # Deterministic tie break by original column index. Validation is untouched.
        order = np.lexsort((np.arange(len(gain)), -gain))[: case.gain_top_k]
        order.sort()
        x_fit, x_valid = x_fit[:, order], x_valid[:, order]
    return x_fit, x_valid, x_fit.shape[1]


def evaluate_case(folds: list[PreparedFold], case: FeatureCase, seed: int = 42) -> dict:
    n_rows = sum(len(f.valid_index) for f in folds)
    prediction = np.empty(n_rows, dtype=object)
    probabilities: np.ndarray | None = None
    classes: np.ndarray | None = None
    fold_scores, feature_counts = [], []
    for fold in folds:
        x_fit, x_valid, feature_count = matrices_for_case(fold, case, seed + fold.fold)
        model = make_lgbm(seed + fold.fold)
        model.fit(x_fit, fold.y_fit)
        if classes is None:
            classes = np.asarray(sorted(np.unique(np.concatenate([f.y_fit for f in folds]))))
            probabilities = np.zeros((n_rows, len(classes)), dtype=np.float32)
        local = model.predict_proba(x_valid)
        aligned = np.zeros((len(fold.valid_index), len(classes)), dtype=np.float32)
        lookup = {name: i for i, name in enumerate(classes)}
        for j, name in enumerate(model.classes_):
            aligned[:, lookup[name]] = local[:, j]
        probabilities[fold.valid_index] = aligned
        fold_pred = classes[np.argmax(aligned, axis=1)]
        prediction[fold.valid_index] = fold_pred
        score = f1_score(fold.y_valid, fold_pred, average="macro", zero_division=0)
        fold_scores.append(score)
        feature_counts.append(feature_count)
        print(f"[{case.name}] fold={fold.fold}/5 f1={score:.6f} features={feature_count:,}")
    truth = np.empty(n_rows, dtype=object)
    for fold in folds:
        truth[fold.valid_index] = fold.y_valid
    result = {
        "case": case.name,
        "description": case.description,
        "seed": seed,
        "oof_f1_macro": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "oof_accuracy": float(accuracy_score(truth, prediction)),
        "fold_f1_mean": float(np.mean(fold_scores)),
        "fold_f1_std": float(np.std(fold_scores)),
        "feature_count_mean": float(np.mean(feature_counts)),
        "prediction": prediction,
        "probabilities": probabilities,
        "classes": classes,
    }
    print(f"=> {case.name}: OOF Macro F1={result['oof_f1_macro']:.6f}")
    return result


def evaluate_group(folds: list[PreparedFold], group: str, seed: int = 42) -> dict[str, dict]:
    return {case.name: evaluate_case(folds, case, seed) for case in cases_for_group(group)}


def leaderboard(results: dict[str, dict]) -> pd.DataFrame:
    rows = [{k: v for k, v in result.items() if k not in {"prediction", "probabilities", "classes"}} for result in results.values()]
    return pd.DataFrame(rows).sort_values("oof_f1_macro", ascending=False).reset_index(drop=True)


def _truth_from_folds(folds: list[PreparedFold]) -> np.ndarray:
    truth = np.empty(sum(len(f.valid_index) for f in folds), dtype=object)
    for fold in folds:
        truth[fold.valid_index] = fold.y_valid
    return truth


def evaluate_lr_reference(folds: list[PreparedFold], seed: int = 42) -> dict:
    """Run the exp13 LR champion on the same prepared full matrices."""

    n_rows = sum(len(f.valid_index) for f in folds)
    classes = np.asarray(sorted(np.unique(np.concatenate([f.y_fit for f in folds]))))
    probabilities = np.zeros((n_rows, len(classes)), dtype=np.float32)
    lookup = {name: i for i, name in enumerate(classes)}
    for fold in folds:
        model = P13.make_model(seed)
        model.fit(fold.x_fit, fold.y_fit)
        local = model.predict_proba(fold.x_valid)
        for j, name in enumerate(model.classes_):
            probabilities[fold.valid_index, lookup[name]] = local[:, j]
    truth = _truth_from_folds(folds)
    prediction = classes[np.argmax(probabilities, axis=1)]
    return {
        "case": "LR_exp13_champion",
        "seed": seed,
        "oof_f1_macro": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "oof_accuracy": float(accuracy_score(truth, prediction)),
        "prediction": prediction,
        "probabilities": probabilities,
        "classes": classes,
    }


def diversity_vs_lr(results: dict[str, dict], lr_result: dict) -> pd.DataFrame:
    rows = []
    lr_prediction = lr_result["prediction"]
    for name, result in results.items():
        prediction = result["prediction"]
        rows.append({
            "case": name,
            "lgbm_f1": result["oof_f1_macro"],
            "lr_f1": lr_result["oof_f1_macro"],
            "prediction_disagreement": float(np.mean(prediction != lr_prediction)),
        })
    return pd.DataFrame(rows).sort_values(["lgbm_f1", "prediction_disagreement"], ascending=False)


def search_lr_blends(
    folds: list[PreparedFold],
    results: dict[str, dict],
    lr_result: dict,
    case_names: list[str],
    lgbm_weights=(0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50),
) -> pd.DataFrame:
    """Blend probabilities; weight denotes the LGBM share."""

    truth = _truth_from_folds(folds)
    classes = lr_result["classes"]
    rows = []
    for name in case_names:
        result = results[name]
        if not np.array_equal(result["classes"], classes):
            raise ValueError(f"class order mismatch: {name}")
        for weight in lgbm_weights:
            probability = (1.0 - weight) * lr_result["probabilities"] + weight * result["probabilities"]
            prediction = classes[np.argmax(probability, axis=1)]
            rows.append({
                "case": name,
                "lgbm_weight": weight,
                "lr_weight": 1.0 - weight,
                "oof_f1_macro": float(f1_score(truth, prediction, average="macro", zero_division=0)),
                "oof_accuracy": float(accuracy_score(truth, prediction)),
                "delta_vs_lr": float(f1_score(truth, prediction, average="macro", zero_division=0) - lr_result["oof_f1_macro"]),
            })
    return pd.DataFrame(rows).sort_values("oof_f1_macro", ascending=False).reset_index(drop=True)


def search_lgbm_pair_blends(
    folds: list[PreparedFold],
    results: dict[str, dict],
    case_names: list[str],
    first_weights=(0.25, 0.50, 0.75),
) -> pd.DataFrame:
    """Check whether two different LGBM feature spaces complement each other."""

    truth = _truth_from_folds(folds)
    rows = []
    for left_pos, left in enumerate(case_names):
        for right in case_names[left_pos + 1:]:
            classes = results[left]["classes"]
            if not np.array_equal(classes, results[right]["classes"]):
                raise ValueError("class order mismatch")
            for weight in first_weights:
                probability = weight * results[left]["probabilities"] + (1.0 - weight) * results[right]["probabilities"]
                prediction = classes[np.argmax(probability, axis=1)]
                rows.append({
                    "left": left,
                    "right": right,
                    "left_weight": weight,
                    "oof_f1_macro": float(f1_score(truth, prediction, average="macro", zero_division=0)),
                    "oof_accuracy": float(accuracy_score(truth, prediction)),
                })
    return pd.DataFrame(rows).sort_values("oof_f1_macro", ascending=False).reset_index(drop=True)


def confirm_cases(train: pd.DataFrame, genes: list[str], case_names: list[str], seeds=(42, 52, 62)) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        folds = prepare_seed(train, genes, seed)
        for name in case_names:
            result = evaluate_case(folds, CASE_MAP[name], seed)
            rows.append({"case": name, "seed": seed, "oof_f1_macro": result["oof_f1_macro"], "oof_accuracy": result["oof_accuracy"], "features": result["feature_count_mean"]})
    return pd.DataFrame(rows)


def confirm_lr_blends(
    train: pd.DataFrame,
    genes: list[str],
    configurations: list[tuple[str, float]],
    seeds=(42, 52, 62),
) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        folds = prepare_seed(train, genes, seed)
        lr_result = evaluate_lr_reference(folds, seed)
        cache: dict[str, dict] = {}
        truth = _truth_from_folds(folds)
        for name, weight in configurations:
            if name not in cache:
                cache[name] = evaluate_case(folds, CASE_MAP[name], seed)
            result = cache[name]
            probability = (1.0 - weight) * lr_result["probabilities"] + weight * result["probabilities"]
            prediction = lr_result["classes"][np.argmax(probability, axis=1)]
            score = f1_score(truth, prediction, average="macro", zero_division=0)
            rows.append({
                "case": name,
                "lgbm_weight": weight,
                "seed": seed,
                "oof_f1_macro": float(score),
                "oof_accuracy": float(accuracy_score(truth, prediction)),
                "lr_f1": lr_result["oof_f1_macro"],
                "delta_vs_lr": float(score - lr_result["oof_f1_macro"]),
            })
    return pd.DataFrame(rows)
