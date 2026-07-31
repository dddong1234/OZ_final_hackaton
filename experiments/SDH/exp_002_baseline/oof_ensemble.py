from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, chi2
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from common.starter_preprocess import WTBinaryEncoder


N_SPLITS = 5
CV_SEED = 42
MODEL_SEED = 42


class MinimumMutationCount(
    TransformerMixin,
    BaseEstimator,
):
    """Keep genes mutated in at least ``min_count`` training samples."""

    def __init__(self, min_count: int = 3) -> None:
        self.min_count = min_count

    def fit(self, X: Any, y: Any = None) -> MinimumMutationCount:
        del y
        if self.min_count < 1:
            raise ValueError("min_count는 1 이상이어야 합니다.")
        mutation_counts = np.asarray(X).sum(axis=0)
        self.support_ = mutation_counts >= self.min_count
        if not np.any(self.support_):
            raise ValueError("MinimumMutationCount가 모든 피처를 제거했습니다.")
        return self

    def transform(self, X: Any) -> Any:
        if not hasattr(self, "support_"):
            raise RuntimeError("transform 전에 fit을 실행해야 합니다.")
        if hasattr(X, "iloc"):
            return X.iloc[:, self.support_]
        return X[:, self.support_]


class MutationBurdenAppender(
    TransformerMixin,
    BaseEstimator,
):
    """Append log1p mutation count to binary gene features."""

    def fit(self, X: Any, y: Any = None) -> MutationBurdenAppender:
        del y
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X: Any) -> np.ndarray:
        values = np.asarray(X, dtype=np.float32)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("fit 시점과 transform 시점의 피처 수가 다릅니다.")
        log_mutation_count = np.log1p(values.sum(axis=1, keepdims=True))
        return np.hstack([values, log_mutation_count]).astype(
            np.float32,
            copy=False,
        )


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    preprocessor: Any
    model_factory: Callable[[], Any]


@dataclass
class EnsembleSprintResult:
    leaderboard: pd.DataFrame
    blend_leaderboard: pd.DataFrame
    oof_probabilities: dict[str, np.ndarray]
    test_probabilities: dict[str, np.ndarray]
    classes: np.ndarray
    best_single: str
    best_blend_weights: dict[str, float]
    submission_paths: dict[str, Path]


def _preprocessor(kind: str) -> Pipeline:
    steps: list[tuple[str, Any]] = [("wt_binary", WTBinaryEncoder())]

    if kind == "baseline":
        pass
    elif kind == "nonconstant":
        steps.append(("remove_constant", VarianceThreshold()))
    elif kind == "min_count_3":
        steps.append(("min_count", MinimumMutationCount(min_count=3)))
    elif kind == "min_count_5":
        steps.append(("min_count", MinimumMutationCount(min_count=5)))
    elif kind == "chi2_1000":
        steps.extend(
            [
                ("remove_constant", VarianceThreshold()),
                ("select", SelectKBest(chi2, k=1000)),
            ]
        )
    elif kind == "chi2_2000":
        steps.extend(
            [
                ("remove_constant", VarianceThreshold()),
                ("select", SelectKBest(chi2, k=2000)),
            ]
        )
    elif kind == "tfidf":
        steps.extend(
            [
                ("remove_constant", VarianceThreshold()),
                (
                    "tfidf",
                    TfidfTransformer(
                        norm="l2",
                        use_idf=True,
                        smooth_idf=True,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
    elif kind == "burden":
        steps.extend(
            [
                ("remove_constant", VarianceThreshold()),
                ("append_burden", MutationBurdenAppender()),
            ]
        )
    else:
        raise ValueError(f"알 수 없는 전처리: {kind}")

    return Pipeline(steps)


def _logistic(c_value: float, class_weight: str | None = "balanced") -> Any:
    return LogisticRegression(
        C=c_value,
        solver="lbfgs",
        max_iter=2000,
        class_weight=class_weight,
        random_state=MODEL_SEED,
    )


def make_candidates(mode: str = "fast") -> list[Candidate]:
    """Create a practical model/preprocessing search set.

    ``fast`` is intended for same-day submission selection. ``full`` expands
    the regularization and preprocessing search when more runtime is available.
    """

    if mode not in {"fast", "full"}:
        raise ValueError("mode는 'fast' 또는 'full'이어야 합니다.")

    specs: list[tuple[str, str, Any, Callable[[], Any]]] = [
        *[
            (
                f"lr_baseline_c{c_value:g}",
                "lr_baseline",
                _preprocessor("baseline"),
                lambda c_value=c_value: _logistic(c_value),
            )
            for c_value in (0.03, 0.1, 0.3, 1.0)
        ],
        *[
            (
                f"lr_nonconstant_c{c_value:g}",
                "lr_nonconstant",
                _preprocessor("nonconstant"),
                lambda c_value=c_value: _logistic(c_value),
            )
            for c_value in (0.1, 0.3)
        ],
        (
            "lr_min_count3_c0.1",
            "lr_min_count",
            _preprocessor("min_count_3"),
            lambda: _logistic(0.1),
        ),
        (
            "lr_chi2_1000_c0.1",
            "lr_chi2",
            _preprocessor("chi2_1000"),
            lambda: _logistic(0.1),
        ),
        (
            "lr_chi2_2000_c0.1",
            "lr_chi2",
            _preprocessor("chi2_2000"),
            lambda: _logistic(0.1),
        ),
        (
            "lr_tfidf_c1",
            "lr_tfidf",
            _preprocessor("tfidf"),
            lambda: _logistic(1.0),
        ),
        (
            "lr_burden_c0.1",
            "lr_burden",
            _preprocessor("burden"),
            lambda: _logistic(0.1),
        ),
        (
            "complement_nb_a1",
            "naive_bayes",
            _preprocessor("nonconstant"),
            lambda: ComplementNB(alpha=1.0),
        ),
        (
            "sgd_log_a0.0001",
            "sgd",
            _preprocessor("tfidf"),
            lambda: SGDClassifier(
                loss="log_loss",
                alpha=1e-4,
                max_iter=3000,
                tol=1e-4,
                class_weight="balanced",
                random_state=MODEL_SEED,
            ),
        ),
    ]

    try:
        from lightgbm import LGBMClassifier

        specs.append(
            (
                "lightgbm_nonconstant",
                "lightgbm",
                _preprocessor("nonconstant"),
                lambda: LGBMClassifier(
                    objective="multiclass",
                    n_estimators=500,
                    learning_rate=0.05,
                    num_leaves=31,
                    class_weight="balanced",
                    random_state=MODEL_SEED,
                    n_jobs=-1,
                    verbosity=-1,
                    deterministic=True,
                    force_col_wise=True,
                ),
            )
        )
    except ImportError:
        print("[skip] lightgbm이 설치되어 있지 않아 후보에서 제외합니다.")

    if mode == "full":
        specs.extend(
            [
                *[
                    (
                        f"lr_baseline_c{c_value:g}_{weight_name}",
                        "lr_baseline",
                        _preprocessor("baseline"),
                        lambda c_value=c_value, weight=weight: _logistic(
                            c_value,
                            weight,
                        ),
                    )
                    for c_value in (0.01, 3.0, 10.0)
                    for weight_name, weight in (
                        ("balanced", "balanced"),
                        ("none", None),
                    )
                ],
                (
                    "lr_min_count5_c0.1",
                    "lr_min_count",
                    _preprocessor("min_count_5"),
                    lambda: _logistic(0.1),
                ),
                (
                    "lr_tfidf_c3",
                    "lr_tfidf",
                    _preprocessor("tfidf"),
                    lambda: _logistic(3.0),
                ),
                (
                    "complement_nb_a0.1",
                    "naive_bayes",
                    _preprocessor("nonconstant"),
                    lambda: ComplementNB(alpha=0.1),
                ),
                (
                    "complement_nb_a10",
                    "naive_bayes",
                    _preprocessor("nonconstant"),
                    lambda: ComplementNB(alpha=10.0),
                ),
                (
                    "sgd_log_a0.00001",
                    "sgd",
                    _preprocessor("tfidf"),
                    lambda: SGDClassifier(
                        loss="log_loss",
                        alpha=1e-5,
                        max_iter=3000,
                        tol=1e-4,
                        class_weight="balanced",
                        random_state=MODEL_SEED,
                    ),
                ),
            ]
        )

    return [
        Candidate(
            name=name,
            family=family,
            preprocessor=preprocessor,
            model_factory=model_factory,
        )
        for name, family, preprocessor, model_factory in specs
    ]


def _align_probabilities(
    probabilities: np.ndarray,
    model_classes: np.ndarray,
    all_classes: np.ndarray,
) -> np.ndarray:
    aligned = np.zeros((len(probabilities), len(all_classes)), dtype=np.float64)
    class_to_index = {
        class_name: index for index, class_name in enumerate(all_classes)
    }
    for source_index, class_name in enumerate(model_classes):
        aligned[:, class_to_index[class_name]] = probabilities[:, source_index]
    return aligned


def _integer_compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)]
    output: list[tuple[int, ...]] = []
    for first in range(total + 1):
        for tail in _integer_compositions(total - first, parts - 1):
            output.append((first, *tail))
    return output


def _select_blend_pool(
    leaderboard: pd.DataFrame,
    max_models: int = 5,
) -> list[str]:
    family_best = (
        leaderboard.sort_values("f1_macro", ascending=False)
        .drop_duplicates("family")
        .head(max_models)
    )
    return family_best["candidate"].tolist()


def _search_blends(
    y: np.ndarray,
    classes: np.ndarray,
    oof_probabilities: dict[str, np.ndarray],
    pool: list[str],
    step: float = 0.1,
) -> pd.DataFrame:
    units = round(1 / step)
    rows: list[dict[str, Any]] = []

    for weights_integer in _integer_compositions(units, len(pool)):
        if sum(weight > 0 for weight in weights_integer) < 2:
            continue
        weights = np.asarray(weights_integer, dtype=float) / units
        blended = sum(
            weight * oof_probabilities[name]
            for name, weight in zip(pool, weights, strict=True)
        )
        prediction = classes[blended.argmax(axis=1)]
        rows.append(
            {
                "f1_macro": f1_score(y, prediction, average="macro"),
                "accuracy": accuracy_score(y, prediction),
                "weights": {
                    name: float(weight)
                    for name, weight in zip(pool, weights, strict=True)
                    if weight > 0
                },
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["f1_macro", "accuracy"],
        ascending=False,
    ).reset_index(drop=True)


def _save_submission(
    sample_submission: pd.DataFrame,
    id_column: str,
    target_column: str,
    prediction: np.ndarray,
    path: Path,
) -> Path:
    output = sample_submission.copy()
    output[target_column] = prediction
    if output[id_column].duplicated().any():
        raise ValueError("submission ID에 중복이 있습니다.")
    if output[target_column].isna().any():
        raise ValueError("submission 예측에 결측값이 있습니다.")
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    return path


def run_oof_ensemble_sprint(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    *,
    output_dir: str | Path,
    mode: str = "fast",
    target_column: str = "SUBCLASS",
    id_column: str = "ID",
) -> EnsembleSprintResult:
    """Evaluate candidates and create three submission options."""

    feature_columns = [
        column
        for column in train.columns
        if column not in {id_column, target_column}
    ]
    test_feature_columns = [
        column for column in test.columns if column != id_column
    ]
    if feature_columns != test_feature_columns:
        raise ValueError("train/test feature 컬럼 또는 순서가 다릅니다.")
    if not test[id_column].equals(sample_submission[id_column]):
        raise ValueError("test와 sample_submission의 ID 또는 순서가 다릅니다.")

    X = train[feature_columns].reset_index(drop=True)
    y = train[target_column].astype(str).to_numpy()
    X_test = test[feature_columns].reset_index(drop=True)
    classes = np.asarray(sorted(np.unique(y)), dtype=object)
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=CV_SEED,
    )

    candidates = make_candidates(mode)
    leaderboard_rows: list[dict[str, Any]] = []
    oof_probabilities: dict[str, np.ndarray] = {}
    test_probabilities: dict[str, np.ndarray] = {}

    print(
        f"후보 {len(candidates)}개 × {N_SPLITS}-Fold 실행 "
        f"(mode={mode}, features={len(feature_columns):,})"
    )

    for candidate_index, candidate in enumerate(candidates, start=1):
        started = perf_counter()
        oof = np.zeros((len(X), len(classes)), dtype=np.float64)
        test_average = np.zeros((len(X_test), len(classes)), dtype=np.float64)
        fold_scores: list[float] = []

        for fold, (train_index, valid_index) in enumerate(
            cv.split(X, y),
            start=1,
        ):
            pipeline = Pipeline(
                [
                    ("preprocessing", clone(candidate.preprocessor)),
                    ("model", candidate.model_factory()),
                ]
            )
            pipeline.fit(X.iloc[train_index], y[train_index])
            model_classes = np.asarray(
                pipeline.named_steps["model"].classes_,
                dtype=object,
            )
            valid_proba = _align_probabilities(
                pipeline.predict_proba(X.iloc[valid_index]),
                model_classes,
                classes,
            )
            test_proba = _align_probabilities(
                pipeline.predict_proba(X_test),
                model_classes,
                classes,
            )
            oof[valid_index] = valid_proba
            test_average += test_proba / N_SPLITS
            fold_prediction = classes[valid_proba.argmax(axis=1)]
            fold_f1 = f1_score(
                y[valid_index],
                fold_prediction,
                average="macro",
            )
            fold_scores.append(fold_f1)
            print(
                f"[{candidate_index:02d}/{len(candidates):02d}] "
                f"{candidate.name} fold={fold}/{N_SPLITS} "
                f"f1={fold_f1:.5f}"
            )

        prediction = classes[oof.argmax(axis=1)]
        oof_f1 = f1_score(y, prediction, average="macro")
        oof_accuracy = accuracy_score(y, prediction)
        elapsed = perf_counter() - started
        oof_probabilities[candidate.name] = oof
        test_probabilities[candidate.name] = test_average
        leaderboard_rows.append(
            {
                "candidate": candidate.name,
                "family": candidate.family,
                "f1_macro": oof_f1,
                "accuracy": oof_accuracy,
                "fold_f1_mean": float(np.mean(fold_scores)),
                "fold_f1_std": float(np.std(fold_scores, ddof=1)),
                "seconds": elapsed,
            }
        )
        print(
            f"  -> OOF Macro F1={oof_f1:.5f}, "
            f"Accuracy={oof_accuracy:.5f}, time={elapsed:.1f}s"
        )

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["f1_macro", "accuracy"],
        ascending=False,
    ).reset_index(drop=True)
    best_single = str(leaderboard.iloc[0]["candidate"])

    blend_pool = _select_blend_pool(leaderboard)
    blend_leaderboard = _search_blends(
        y,
        classes,
        oof_probabilities,
        blend_pool,
    )
    best_blend_weights = dict(blend_leaderboard.iloc[0]["weights"])

    best_blend_test = sum(
        weight * test_probabilities[name]
        for name, weight in best_blend_weights.items()
    )
    top_three = leaderboard.head(3)["candidate"].tolist()
    equal_test = np.mean(
        [test_probabilities[name] for name in top_three],
        axis=0,
    )

    output_dir = Path(output_dir)
    submission_paths = {
        "best_single": _save_submission(
            sample_submission,
            id_column,
            target_column,
            classes[test_probabilities[best_single].argmax(axis=1)],
            output_dir / "submission_oof_best_single.csv",
        ),
        "best_soft_vote": _save_submission(
            sample_submission,
            id_column,
            target_column,
            classes[best_blend_test.argmax(axis=1)],
            output_dir / "submission_oof_best_soft_vote.csv",
        ),
        "top3_equal_vote": _save_submission(
            sample_submission,
            id_column,
            target_column,
            classes[equal_test.argmax(axis=1)],
            output_dir / "submission_oof_top3_equal_vote.csv",
        ),
    }

    print("\n=== 단일 모델 순위 ===")
    print(leaderboard.to_string(index=False))
    print("\n=== 최고 soft-voting ===")
    print(blend_leaderboard.head(10).to_string(index=False))
    print("\n저장된 제출 파일:")
    for name, path in submission_paths.items():
        print(f"- {name}: {path}")

    return EnsembleSprintResult(
        leaderboard=leaderboard,
        blend_leaderboard=blend_leaderboard,
        oof_probabilities=oof_probabilities,
        test_probabilities=test_probabilities,
        classes=classes,
        best_single=best_single,
        best_blend_weights=best_blend_weights,
        submission_paths=submission_paths,
    )
