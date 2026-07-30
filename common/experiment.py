from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MODEL_METHODS = ("fit", "predict", "predict_proba")


def load_config(member: str, experiment: str) -> dict[str, Any]:
    common_path = PROJECT_ROOT / "configs" / "baseline.yaml"
    member_path = (
        PROJECT_ROOT / "experiments" / member / experiment / "config.yaml"
    )
    common_config = yaml.safe_load(common_path.read_text(encoding="utf-8"))
    member_config = yaml.safe_load(member_path.read_text(encoding="utf-8"))
    for section, values in member_config.items():
        if isinstance(values, dict) and isinstance(common_config.get(section), dict):
            common_config[section].update(values)
        else:
            common_config[section] = values
    return common_config


def validate_model_interface(model: Any) -> None:
    missing = [
        method
        for method in REQUIRED_MODEL_METHODS
        if not callable(getattr(model, method, None))
    ]
    if missing:
        raise TypeError(
            "모델에 필요한 메서드가 없습니다: "
            + ", ".join(missing)
            + ". training/model.py의 build_model() 반환값을 확인하세요."
        )


def aligned_probabilities(model: Any, features: Any, classes: np.ndarray) -> np.ndarray:
    if not hasattr(model, "classes_"):
        raise TypeError("학습된 모델에 classes_ 속성이 없습니다.")
    model_classes = np.asarray(model.classes_)
    if set(model_classes) != set(classes):
        raise ValueError("앙상블 모델의 클래스 구성이 서로 다릅니다.")
    positions = [int(np.where(model_classes == label)[0][0]) for label in classes]
    return np.asarray(model.predict_proba(features))[:, positions]


def run_eda(member: str) -> None:
    common_path = PROJECT_ROOT / "configs" / "baseline.yaml"
    config = yaml.safe_load(common_path.read_text(encoding="utf-8"))
    data_config = config["data"]
    train = pd.read_csv(PROJECT_ROOT / data_config["train_path"])
    test = pd.read_csv(PROJECT_ROOT / data_config["test_path"])
    target = data_config["target_column"]
    results_dir = PROJECT_ROOT / "experiments" / member / "notebooks"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_columns": len(train.columns),
        "test_columns": len(test.columns),
        "target": target,
        "class_count": int(train[target].nunique()),
        "train_missing_cells": int(train.isna().sum().sum()),
        "test_missing_cells": int(test.isna().sum().sum()),
    }
    (results_dir / "eda_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    train[target].value_counts().rename_axis(target).reset_index(
        name="count"
    ).to_csv(results_dir / "class_distribution.csv", index=False)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"EDA 결과 저장: {results_dir}")


def run_training(member: str, experiment: str) -> None:
    config = load_config(member, experiment)
    data_config = config["data"]
    validation_config = config["validation"]
    target = data_config["target_column"]
    id_column = data_config["id_column"]
    results_dir = (
        PROJECT_ROOT / "experiments" / member / experiment / "results"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(PROJECT_ROOT / data_config["train_path"])
    test = pd.read_csv(PROJECT_ROOT / data_config["test_path"])
    submission = pd.read_csv(PROJECT_ROOT / data_config["submission_path"])

    preprocessing = importlib.import_module(
        f"experiments.{member}.{experiment}.preprocessing.preprocess"
    )
    model_factory = importlib.import_module(
        f"experiments.{member}.{experiment}.training.model"
    )
    seeds = config["experiment"].get("seeds")
    if not seeds:
        seeds = [config["experiment"].get("seed", 42)]

    seed_metrics = []
    trained_models = []
    probability_sum = None
    class_order = None

    for seed in seeds:
        train_part, valid_part = train_test_split(
            train,
            test_size=validation_config["test_size"],
            random_state=seed,
            stratify=train[target] if validation_config["stratify"] else None,
        )
        state = preprocessing.fit(train_part, target, id_column)
        x_train = preprocessing.transform(train_part, state, target, id_column)
        x_valid = preprocessing.transform(valid_part, state, target, id_column)
        x_test = preprocessing.transform(test, state, target, id_column)

        model = model_factory.build_model(config["model"], seed)
        validate_model_interface(model)
        model.fit(x_train, train_part[target])
        valid_predictions = model.predict(x_valid)
        seed_metrics.append(
            {
                "seed": seed,
                "accuracy": float(
                    accuracy_score(valid_part[target], valid_predictions)
                ),
                "f1_macro": float(
                    f1_score(
                        valid_part[target],
                        valid_predictions,
                        average="macro",
                    )
                ),
            }
        )
        if probability_sum is None:
            if not hasattr(model, "classes_"):
                raise TypeError("학습된 모델에 classes_ 속성이 없습니다.")
            class_order = np.asarray(model.classes_)
            probability_sum = aligned_probabilities(model, x_test, class_order)
        else:
            probability_sum += aligned_probabilities(model, x_test, class_order)
        trained_models.append(
            {"seed": seed, "model": model, "preprocess_state": state}
        )

    metrics = {
        "experiment": config["experiment"]["id"],
        "owner": member,
        "seeds": seeds,
        "validation": validation_config,
        "accuracy_mean": float(
            np.mean([item["accuracy"] for item in seed_metrics])
        ),
        "f1_macro_mean": float(
            np.mean([item["f1_macro"] for item in seed_metrics])
        ),
        "per_seed": seed_metrics,
    }

    test_predictions = class_order[np.argmax(probability_sum, axis=1)]
    submission[id_column] = test[id_column].values
    submission[target] = test_predictions
    submission.to_csv(results_dir / "submission.csv", index=False)
    (results_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    joblib.dump(
        {
            "member": member,
            "experiment": experiment,
            "classes": class_order,
            "members": trained_models,
            "ensemble": "mean_probability",
        },
        results_dir / "model.joblib",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"학습 결과 저장: {results_dir}")


def run_inference(member: str, experiment: str) -> None:
    config = load_config(member, experiment)
    data_config = config["data"]
    target = data_config["target_column"]
    id_column = data_config["id_column"]
    results_dir = (
        PROJECT_ROOT / "experiments" / member / experiment / "results"
    )
    model_path = results_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"저장된 모델이 없습니다: {model_path}\n"
            "먼저 training.run을 실행하세요."
        )

    test = pd.read_csv(PROJECT_ROOT / data_config["test_path"])
    submission = pd.read_csv(PROJECT_ROOT / data_config["submission_path"])
    if len(test) != len(submission):
        raise ValueError("test와 sample_submission의 행 수가 다릅니다.")

    preprocessing = importlib.import_module(
        f"experiments.{member}.{experiment}.preprocessing.preprocess"
    )
    bundle = joblib.load(model_path)
    if bundle.get("member") != member or bundle.get("experiment") != experiment:
        raise ValueError("저장 모델과 현재 실험 경로가 일치하지 않습니다.")

    classes = np.asarray(bundle["classes"])
    probability_sum = None
    for saved in bundle["members"]:
        model = saved["model"]
        validate_model_interface(model)
        x_test = preprocessing.transform(
            test,
            saved["preprocess_state"],
            target,
            id_column,
        )
        probabilities = aligned_probabilities(model, x_test, classes)
        probability_sum = (
            probabilities
            if probability_sum is None
            else probability_sum + probabilities
        )

    predictions = classes[np.argmax(probability_sum, axis=1)]
    submission[id_column] = test[id_column].values
    submission[target] = predictions
    output_path = results_dir / "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"추론 완료: {output_path}")
