"""Notebook-friendly benchmark for comparing preprocessing pipelines.

The benchmark owns cross-validation, models, metrics, and OOF generation.
Experiment authors only provide a scikit-learn compatible Transformer or
preprocessing ``Pipeline``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


BenchmarkModel = Literal["logistic", "lightgbm"]

BENCHMARK_N_SPLITS = 5
BENCHMARK_CV_SEED = 42
CONFIRMATION_CV_SEEDS = (42, 52, 62)
MODEL_SEED = 42

LOGISTIC_PARAMS: dict[str, Any] = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": MODEL_SEED,
}

LIGHTGBM_PARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "class_weight": "balanced",
    "random_state": MODEL_SEED,
    "n_jobs": -1,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}


@dataclass(frozen=True)
class BenchmarkResult:
    """Results returned to a notebook after a benchmark run."""

    summary: dict[str, Any]
    run_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    oof_predictions: pd.DataFrame

    def summary_frame(self) -> pd.DataFrame:
        """Return the main result as a compact one-row table."""

        return pd.DataFrame([self.summary])

    def to_metrics_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable metrics record."""

        return {
            **self.summary,
            "per_seed": self.run_metrics.to_dict(orient="records"),
            "per_fold": self.fold_metrics.to_dict(orient="records"),
        }

    def save_metrics(self, path: str | Path) -> Path:
        """Save lightweight metrics only; OOF predictions are not persisted."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                self.to_metrics_dict(),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return output_path


def _build_model(model_name: BenchmarkModel) -> Any:
    if model_name == "logistic":
        return LogisticRegression(**LOGISTIC_PARAMS)
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as error:
            raise ImportError(
                "LightGBM 2차 검증을 실행하려면 requirements.txt의 "
                "lightgbm을 설치하세요."
            ) from error
        return LGBMClassifier(**LIGHTGBM_PARAMS)
    raise ValueError("model은 'logistic' 또는 'lightgbm'이어야 합니다.")


def _model_params(model_name: BenchmarkModel) -> dict[str, Any]:
    if model_name == "logistic":
        return dict(LOGISTIC_PARAMS)
    if model_name == "lightgbm":
        return dict(LIGHTGBM_PARAMS)
    raise ValueError("model은 'logistic' 또는 'lightgbm'이어야 합니다.")


def _stable_parameter_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if callable(value):
        module = getattr(value, "__module__", value.__class__.__module__)
        name = getattr(value, "__qualname__", value.__class__.__qualname__)
        return f"{module}.{name}"
    if isinstance(value, (list, tuple)):
        return [_stable_parameter_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _stable_parameter_value(item)
            for key, item in value.items()
        }
    if callable(getattr(value, "get_params", None)):
        return f"{value.__class__.__module__}.{value.__class__.__name__}"
    return f"{value.__class__.__module__}.{value.__class__.__name__}"


def _json_safe_parameters(estimator: Any) -> dict[str, Any]:
    return {
        name: _stable_parameter_value(value)
        for name, value in estimator.get_params(deep=True).items()
    }


def _validate_preprocessor(preprocessor: Any) -> None:
    missing_methods = [
        method
        for method in ("fit", "transform", "get_params")
        if not callable(getattr(preprocessor, method, None))
    ]
    if missing_methods:
        raise TypeError(
            "preprocessor는 sklearn Transformer 또는 Pipeline이어야 합니다. "
            f"누락된 메서드: {missing_methods}"
        )
    try:
        clone(preprocessor)
    except Exception as error:
        raise TypeError(
            "preprocessor를 sklearn.base.clone으로 복제할 수 없습니다. "
            "커스텀 Transformer는 __init__ 파라미터를 그대로 속성에 "
            "저장해야 합니다."
        ) from error


def _validate_source_data(
    train_df: pd.DataFrame,
    target_column: str,
    id_column: str,
) -> None:
    missing_columns = [
        column
        for column in (id_column, target_column)
        if column not in train_df.columns
    ]
    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")
    if train_df.empty:
        raise ValueError("train_df가 비어 있습니다.")
    if train_df[id_column].isna().any():
        raise ValueError(f"{id_column}에 결측값이 있습니다.")
    if train_df[id_column].duplicated().any():
        raise ValueError(f"{id_column}에 중복값이 있습니다.")
    if train_df[target_column].isna().any():
        raise ValueError(f"{target_column}에 결측값이 있습니다.")

    minimum_class_size = int(train_df[target_column].value_counts().min())
    if minimum_class_size < BENCHMARK_N_SPLITS:
        raise ValueError(
            "StratifiedKFold-5를 적용하려면 모든 클래스에 최소 5개 "
            f"샘플이 필요합니다. 현재 최소 클래스 크기: {minimum_class_size}"
        )


def _aligned_probabilities(
    model: Any,
    features: Any,
    classes: np.ndarray,
) -> np.ndarray:
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise TypeError("벤치마크 모델은 predict_proba와 classes_를 제공해야 합니다.")

    model_classes = np.asarray(model.classes_)
    if set(model_classes) != set(classes):
        raise ValueError("fold 모델과 전체 데이터의 클래스 구성이 다릅니다.")
    positions = [int(np.where(model_classes == label)[0][0]) for label in classes]
    return np.asarray(model.predict_proba(features))[:, positions]


def run_preprocessing_benchmark(
    train_df: pd.DataFrame,
    preprocessor: Any,
    *,
    experiment_id: str,
    preprocessing_name: str | None = None,
    model: BenchmarkModel = "logistic",
    confirmation: bool = False,
    target_column: str = "SUBCLASS",
    id_column: str = "ID",
    verbose: bool = True,
) -> BenchmarkResult:
    """Evaluate a supplied sklearn preprocessor under fixed team conditions.

    Parameters
    ----------
    train_df:
        Raw training dataframe containing ID, target, and feature columns.
    preprocessor:
        A scikit-learn compatible Transformer or preprocessing Pipeline.
        It is cloned and fitted independently inside every CV fold.
    experiment_id:
        Identifier stored in the returned metrics.
    preprocessing_name:
        Human-readable preprocessing name. Defaults to ``experiment_id``.
    model:
        ``"logistic"`` for the primary benchmark or ``"lightgbm"`` for
        secondary nonlinear verification.
    confirmation:
        False uses the canonical seed 42. True repeats 5-fold CV with
        seeds 42, 52, and 62.

    Notes
    -----
    The function intentionally does not expose fold count, model parameters,
    or benchmark seeds. This keeps comparisons identical across team members.
    """

    _validate_source_data(train_df, target_column, id_column)
    _validate_preprocessor(preprocessor)
    if not experiment_id.strip():
        raise ValueError("experiment_id는 비어 있을 수 없습니다.")

    cv_seeds = (
        CONFIRMATION_CV_SEEDS
        if confirmation
        else (BENCHMARK_CV_SEED,)
    )
    classes = np.asarray(
        sorted(train_df[target_column].unique().tolist())
    )
    y = train_df[target_column].reset_index(drop=True)
    source = train_df.reset_index(drop=True)
    features = source.drop(columns=[id_column, target_column])

    run_records: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    oof_frames: list[pd.DataFrame] = []

    benchmark_start = perf_counter()
    for cv_seed in cv_seeds:
        splitter = StratifiedKFold(
            n_splits=BENCHMARK_N_SPLITS,
            shuffle=True,
            random_state=cv_seed,
        )
        oof_pred = np.empty(len(source), dtype=object)
        oof_prob = np.zeros((len(source), len(classes)), dtype=np.float64)
        oof_fold = np.full(len(source), -1, dtype=np.int8)
        run_start = perf_counter()

        for fold, (train_index, valid_index) in enumerate(
            splitter.split(source, y)
        ):
            fold_start = perf_counter()
            x_train = features.iloc[train_index].copy()
            x_valid = features.iloc[valid_index].copy()
            y_train = y.iloc[train_index]

            benchmark_pipeline = Pipeline(
                [
                    ("preprocessing", clone(preprocessor)),
                    ("model", _build_model(model)),
                ]
            )
            benchmark_pipeline.fit(x_train, y_train)
            valid_pred = np.asarray(benchmark_pipeline.predict(x_valid))
            benchmark_model = benchmark_pipeline.named_steps["model"]
            feature_count = int(benchmark_model.n_features_in_)
            valid_prob = _aligned_probabilities(
                benchmark_pipeline,
                x_valid,
                classes,
            )

            oof_pred[valid_index] = valid_pred
            oof_prob[valid_index] = valid_prob
            oof_fold[valid_index] = fold

            fold_record = {
                "cv_seed": int(cv_seed),
                "fold": int(fold),
                "train_rows": int(len(train_index)),
                "valid_rows": int(len(valid_index)),
                "feature_count": feature_count,
                "accuracy": float(
                    accuracy_score(y.iloc[valid_index], valid_pred)
                ),
                "f1_macro": float(
                    f1_score(
                        y.iloc[valid_index],
                        valid_pred,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "elapsed_seconds": float(perf_counter() - fold_start),
            }
            fold_records.append(fold_record)
            if verbose:
                print(
                    f"[{model}] seed={cv_seed} fold={fold + 1}/"
                    f"{BENCHMARK_N_SPLITS} "
                    f"f1_macro={fold_record['f1_macro']:.5f} "
                    f"features={feature_count:,} "
                    f"time={fold_record['elapsed_seconds']:.1f}s"
                )

        if np.any(oof_fold < 0):
            raise RuntimeError("일부 행에 OOF 예측이 생성되지 않았습니다.")

        run_record = {
            "cv_seed": int(cv_seed),
            "oof_accuracy": float(accuracy_score(y, oof_pred)),
            "oof_f1_macro": float(
                f1_score(
                    y,
                    oof_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
            "elapsed_seconds": float(perf_counter() - run_start),
        }
        run_records.append(run_record)

        oof_frame = pd.DataFrame(
            {
                id_column: source[id_column].to_numpy(),
                target_column: y.to_numpy(),
                "cv_seed": int(cv_seed),
                "fold": oof_fold,
                "prediction": oof_pred,
            }
        )
        for class_position, class_name in enumerate(classes):
            oof_frame[str(class_name)] = oof_prob[:, class_position]
        oof_frames.append(oof_frame)

        if verbose:
            print(
                f"[{model}] seed={cv_seed} "
                f"OOF Macro F1={run_record['oof_f1_macro']:.5f}, "
                f"Accuracy={run_record['oof_accuracy']:.5f}"
            )

    run_metrics = pd.DataFrame(run_records)
    fold_metrics = pd.DataFrame(fold_records)
    oof_predictions = pd.concat(oof_frames, ignore_index=True)
    repeated_cv = len(cv_seeds) > 1

    summary = {
        "experiment": experiment_id,
        "preprocessing": preprocessing_name or experiment_id,
        "model": model,
        "validation": {
            "method": "StratifiedKFold",
            "n_splits": BENCHMARK_N_SPLITS,
            "shuffle": True,
            "seeds": list(cv_seeds),
        },
        "model_parameters": _model_params(model),
        "preprocessor_class": (
            f"{preprocessor.__class__.__module__}."
            f"{preprocessor.__class__.__name__}"
        ),
        "preprocessor_parameters": _json_safe_parameters(preprocessor),
        "primary_metric": "oof_f1_macro",
        "oof_f1_macro_mean": float(run_metrics["oof_f1_macro"].mean()),
        "oof_f1_macro_std": (
            float(run_metrics["oof_f1_macro"].std(ddof=1))
            if repeated_cv
            else None
        ),
        "oof_accuracy_mean": float(run_metrics["oof_accuracy"].mean()),
        "oof_accuracy_std": (
            float(run_metrics["oof_accuracy"].std(ddof=1))
            if repeated_cv
            else None
        ),
        "fold_f1_macro_mean": float(fold_metrics["f1_macro"].mean()),
        "fold_f1_macro_std": float(
            fold_metrics["f1_macro"].std(ddof=1)
        ),
        "fold_accuracy_mean": float(fold_metrics["accuracy"].mean()),
        "fold_accuracy_std": float(
            fold_metrics["accuracy"].std(ddof=1)
        ),
        "elapsed_seconds": float(perf_counter() - benchmark_start),
    }

    if verbose:
        if repeated_cv:
            print(
                f"완료: seed별 OOF Macro F1 "
                f"{summary['oof_f1_macro_mean']:.5f} "
                f"± {summary['oof_f1_macro_std']:.5f}"
            )
        else:
            print(
                f"완료: OOF Macro F1 "
                f"{summary['oof_f1_macro_mean']:.5f}"
            )
        print(
            f"fold Macro F1 "
            f"{summary['fold_f1_macro_mean']:.5f} "
            f"± {summary['fold_f1_macro_std']:.5f}"
        )

    return BenchmarkResult(
        summary=summary,
        run_metrics=run_metrics,
        fold_metrics=fold_metrics,
        oof_predictions=oof_predictions,
    )
