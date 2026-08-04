"""앙상블 후보의 예측 다양성 측정 — 슬롯을 쓰기 전에 천장을 본다.

    .venv/bin/python experiments/iljun/exp_011_model_ensemble/run_diversity.py

왜 먼저 재는가
  팀 앙상블 기록이 갈린다.
    ENS-004a  LR + LGBM (GBV)        CV +0.01621 (3/3, σ 초과)   LB 미확인
    gs-002-13 08 + OVR TF-IDF        CV 0.48313 > 08 0.47908     LB 0.38672 < 0.38711
  즉 "CV 가 오르니 앙상블하자"는 이 팀에서 이미 한 번 배신당했다.

  협업 규정 1 이 이유를 미리 적어뒀다 — "LGBM/XGB/CatBoost 를 나누어 맡으면
  예측값 상관이 0.99 를 넘어 앙상블 이득이 사라진다. 각자 다른 천장을 뚫어야
  블렌드에서 시너지가 난다."  그러니 **다양성을 먼저 재고** 붙일지 정한다.

무엇을 재는가 (전부 같은 fold·같은 피처 위에서)
  1. 후보별 단독 성능
  2. 쌍별 예측 불일치율 — 얼마나 다른 답을 내는가
  3. 오라클 상한 — 둘 중 하나라도 맞히는 비율 (= 앙상블 이득의 천장)
  4. 상보성 — 챔피언이 틀린 것 중 상대가 맞히는 비율 (가장 결정적인 수치)
  5. 소프트보팅 가중치 스윕 — 재학습 없이 확률만 섞어 본 이득

후보
  champion  LR multinomial (SDH exp_012 case_04_shrink10 그대로)
  ovr       같은 피처, One-vs-Rest LR   (gs 가 TF-IDF 에서 41% 불일치를 관측한 축)
  lgbm      같은 피처, LightGBM 100 trees (ENS-004a 에서 300 보다 100 이 나았음)

주의
  가중치 스윕은 같은 OOF 에서 최적 w 를 고르므로 낙관 편향이 있다.  채택하려면
  w 를 고정한 뒤 3-seed 로 다시 재야 한다.  이 스크립트는 **후보를 고르는 용도**다.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier

CHAMPION_CASE = "case_04_shrink10"
WEIGHTS = (0.1, 0.2, 0.3, 0.4, 0.5)   # 상대 모델 쪽 가중치


def find_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "data" / "raw" / "train.csv").exists():
            return path
    raise FileNotFoundError("data/raw/train.csv 를 찾지 못했습니다")


def load_sdh(root: Path):
    source = root / "experiments" / "SDH" / "exp_012_enrichment_stability" / "preprocessing.py"
    spec = importlib.util.spec_from_file_location("sdh_exp012_preprocessing", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules["sdh_exp012_preprocessing"] = module
    spec.loader.exec_module(module)
    return module


def make_candidates(seed: int) -> dict:
    """전부 같은 피처 행렬 위에서 학습한다 — 차이는 오직 모델 계열."""
    base = dict(solver="lbfgs", C=0.07, max_iter=2000,
                class_weight="balanced", random_state=seed)
    return {
        "champion": LogisticRegression(**base),
        "ovr": OneVsRestClassifier(LogisticRegression(**base), n_jobs=1),
        "lgbm": LGBMClassifier(objective="multiclass", n_estimators=100,
                               learning_rate=0.05, num_leaves=31,
                               class_weight="balanced", random_state=seed,
                               n_jobs=-1, deterministic=True,
                               force_col_wise=True, verbosity=-1),
    }


def collect_oof(sdh, context, labels: pd.Series, seed: int) -> tuple[dict, np.ndarray]:
    classes = np.unique(labels.to_numpy())
    case = sdh.make_cases()[CHAMPION_CASE]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = {name: np.zeros((len(labels), len(classes))) for name in make_candidates(seed)}

    for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(labels)), labels.to_numpy()), 1):
        # 피처는 한 번만 만들고 세 모델이 공유한다 — 차이를 모델 계열로만 한정한다.
        train_matrix, valid_matrix, _, meta = sdh.build_case_matrices(
            context, tr, va, labels, case, inner_seed=seed)
        print(f"    fold {fold}/5  피처 {meta['total_feature_count']:,}", flush=True)
        for name, model in make_candidates(seed).items():
            started = perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(train_matrix, labels.iloc[tr])
            assert list(model.classes_) == list(classes)
            proba[name][va] = model.predict_proba(valid_matrix)
            print(f"        {name:10s} {(perf_counter()-started):5.0f}s", flush=True)
    return proba, classes


def summarise(proba: dict, classes: np.ndarray, truth: np.ndarray) -> dict:
    predictions = {name: classes[p.argmax(axis=1)] for name, p in proba.items()}
    correct = {name: (pred == truth) for name, pred in predictions.items()}

    singles = {
        name: {"macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
               "accuracy": float(accuracy_score(truth, pred))}
        for name, pred in predictions.items()
    }

    pairs = []
    for other in [n for n in proba if n != "champion"]:
        champion_wrong = ~correct["champion"]
        rescued = int((champion_wrong & correct[other]).sum())
        blends = []
        for weight in WEIGHTS:
            mixed = (1 - weight) * proba["champion"] + weight * proba[other]
            blended = classes[mixed.argmax(axis=1)]
            blends.append({
                "weight_other": weight,
                "macro_f1": float(f1_score(truth, blended, average="macro", zero_division=0)),
            })
        best = max(blends, key=lambda row: row["macro_f1"])
        pairs.append({
            "pair": f"champion + {other}",
            "disagreement_rate": float((predictions["champion"] != predictions[other]).mean()),
            "oracle_either_correct": float((correct["champion"] | correct[other]).mean()),
            "champion_wrong_other_right": rescued,
            "champion_wrong_total": int(champion_wrong.sum()),
            "rescue_rate": float(rescued / max(champion_wrong.sum(), 1)),
            "blend_sweep": blends,
            "best_blend": best,
            "best_blend_gain": best["macro_f1"] - singles["champion"]["macro_f1"],
        })
    return {"singles": singles, "pairs": pairs,
            "champion_top1_accuracy": singles["champion"]["accuracy"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "artifacts" / "diversity.json")
    args = parser.parse_args(argv)

    root = find_root(Path(__file__).resolve())
    sdh = load_sdh(root)
    train = pd.read_csv(root / "data" / "raw" / "train.csv")
    genes = [c for c in train.columns if c not in ("ID", "SUBCLASS")]
    labels = train["SUBCLASS"]
    context = sdh.make_context(train[genes], genes, show_progress=True)

    print(f"\n  seed {args.seed} — 같은 fold·같은 피처, 모델만 교체", flush=True)
    proba, classes = collect_oof(sdh, context, labels, args.seed)
    report = summarise(proba, classes, labels.to_numpy())
    report["seed"] = args.seed
    report["case"] = CHAMPION_CASE
    report["note"] = ("가중치 스윕은 같은 OOF 에서 최적 w 를 골라 낙관 편향이 있다. "
                      "채택하려면 w 고정 후 3-seed 재측정이 필요하다.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"{'후보 단독':16s}{'Macro F1':>12s}{'Accuracy':>12s}")
    print("-" * 70)
    for name, row in report["singles"].items():
        print(f"{name:16s}{row['macro_f1']:>12.5f}{row['accuracy']:>12.5f}")
    print("\n" + "-" * 70)
    print(f"{'쌍':22s}{'불일치':>9s}{'오라클':>9s}{'구제율':>9s}{'최적 blend':>13s}{'이득':>9s}")
    print("-" * 70)
    for pair in report["pairs"]:
        print(f"{pair['pair']:22s}{pair['disagreement_rate']:>9.1%}"
              f"{pair['oracle_either_correct']:>9.1%}{pair['rescue_rate']:>9.1%}"
              f"{pair['best_blend']['macro_f1']:>13.5f}{pair['best_blend_gain']:>+9.5f}"
              f"  (w={pair['best_blend']['weight_other']})")
    print("=" * 70)
    print("불일치 = 두 모델이 다른 답을 낸 비율 | 오라클 = 둘 중 하나라도 맞힌 비율")
    print("구제율 = 챔피언이 틀린 것 중 상대가 맞힌 비율 ← 가장 결정적")
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
