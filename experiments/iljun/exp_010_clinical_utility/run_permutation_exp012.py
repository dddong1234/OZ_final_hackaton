"""exp_012 챔피언의 supervised FE 누수 감사 — SDH 실제 코드 기준.

    .venv/bin/python experiments/iljun/exp_010_clinical_utility/run_permutation_exp012.py

앞선 감사(exp_009 `run_permutation_check.py`)와의 차이
  그때는 exp_011 코드가 레포에 없어 TEAM_REPORT.md 의 수식을 보고 **재구현**해 쟀다.
  그래서 "검증한 것은 실제 코드가 아니라 문서에 적힌 절차"라는 한계를 명시했었다.

  PR #27/#28 로 SDH 코드가 들어왔으므로 이제 그 한계를 닫는다.  이 스크립트는
  `experiments/SDH/exp_012_enrichment_stability/preprocessing.py` 의
  `make_context` / `make_cases` / `build_case_matrices` 를 **그대로 호출**한다.
  enrichment 계산·중첩 cross-fit·표준화 전부 SDH 구현이다.

무엇을 재는가
  label 을 무작위로 섞고 같은 파이프라인을 그대로 돌린다.  섞인 label 에는 실제
  신호가 없으므로 OOF Macro F1 은 우연 수준(26클래스, 약 0.038)이어야 한다.

    핵심 비교 — permutation 하에서 enrichment 가 B04 보다 높은가?

  높다면 cross-fit 이 깨져 모델이 섞인 label 을 되찾고 있다는 뜻이다(= 누수).

대상
  case_00_b04            B04 기준선 (enrichment 없음)
  case_04_shrink10       현재 챔피언 (3-seed 0.52824 ± 0.00187)
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

CHAMPION_CASE = "case_04_shrink10"
BASE_CASE = "case_00_b04"


def find_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "data" / "raw" / "train.csv").exists():
            return path
    raise FileNotFoundError("data/raw/train.csv 를 찾지 못했습니다")


def load_sdh(root: Path):
    """SDH exp_012 preprocessing 모듈을 그대로 로드한다(재구현 아님)."""
    source = root / "experiments" / "SDH" / "exp_012_enrichment_stability" / "preprocessing.py"
    if not source.exists():
        raise FileNotFoundError(f"exp_012 preprocessing 을 찾지 못했습니다: {source}")
    spec = importlib.util.spec_from_file_location("sdh_exp012_preprocessing", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules["sdh_exp012_preprocessing"] = module
    spec.loader.exec_module(module)
    return module


def run_case(sdh, context, labels: pd.Series, case, seed: int, tag: str) -> dict:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    values = labels.to_numpy()
    predicted = np.empty(len(labels), dtype=object)
    warnings_seen = 0
    feature_counts: list[int] = []
    started = perf_counter()

    for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(labels)), values), 1):
        train_matrix, valid_matrix, names, meta = sdh.build_case_matrices(
            context, tr, va, labels, case, inner_seed=seed
        )
        model = sdh.B04.make_model("logistic", seed, None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train_matrix, labels.iloc[tr])
        predicted[va] = model.predict(valid_matrix)
        warnings_seen += sum(issubclass(i.category, ConvergenceWarning) for i in caught)
        feature_counts.append(meta["total_feature_count"])
        print(f"      fold {fold}/5  피처 {meta['total_feature_count']:,} "
              f"(base {meta['base_feature_count']:,} + enrich {meta['extra_feature_count']})",
              flush=True)

    score = float(f1_score(values, predicted, average="macro", zero_division=0))
    print(f"    → {tag}: OOF Macro F1 {score:.6f}  ({(perf_counter()-started)/60:.1f}분)\n",
          flush=True)
    return {"tag": tag, "case": case.name, "oof_macro_f1": score,
            "feature_count_mean": float(np.mean(feature_counts)),
            "convergence_warning_count": warnings_seen}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutation-seed", type=int, default=20260803)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "artifacts" / "permutation_exp012.json")
    args = parser.parse_args(argv)

    root = find_root(Path(__file__).resolve())
    sdh = load_sdh(root)
    cases = sdh.make_cases()

    train = pd.read_csv(root / "data" / "raw" / "train.csv")
    genes = [c for c in train.columns if c not in ("ID", "SUBCLASS")]
    labels = train["SUBCLASS"]
    # test 는 읽지 않는다 — CV 에 필요 없고, 안 읽는 것이 가장 강한 누수 방어다.
    context = sdh.make_context(train[genes], genes, show_progress=True)

    report = {"seed": args.seed, "permutation_seed": args.permutation_seed,
              "source": "experiments/SDH/exp_012_enrichment_stability/preprocessing.py",
              "n_classes": int(labels.nunique()), "runs": {}}

    print("\n[1] 실제 label — SDH 코드를 제 하니스로 몰아 재현되는가", flush=True)
    report["runs"]["real_b04"] = run_case(
        sdh, context, labels, cases[BASE_CASE], args.seed, "real + B04")
    report["runs"]["real_champion"] = run_case(
        sdh, context, labels, cases[CHAMPION_CASE], args.seed, "real + shrink10")

    rng = np.random.default_rng(args.permutation_seed)
    shuffled = pd.Series(labels.to_numpy()[rng.permutation(len(labels))], name="SUBCLASS")
    print(f"[2] label 무작위 셔플 (permutation_seed={args.permutation_seed})", flush=True)
    report["runs"]["permuted_b04"] = run_case(
        sdh, context, shuffled, cases[BASE_CASE], args.seed, "permuted + B04")
    report["runs"]["permuted_champion"] = run_case(
        sdh, context, shuffled, cases[CHAMPION_CASE], args.seed, "permuted + shrink10")

    real_delta = (report["runs"]["real_champion"]["oof_macro_f1"]
                  - report["runs"]["real_b04"]["oof_macro_f1"])
    permuted_delta = (report["runs"]["permuted_champion"]["oof_macro_f1"]
                      - report["runs"]["permuted_b04"]["oof_macro_f1"])
    chance = 1.0 / report["n_classes"]
    report.update({"real_delta": real_delta, "permuted_delta": permuted_delta,
                   "chance_level": chance,
                   "reference": {"sdh_b04_seed42": 0.47786, "sdh_shrink10_seed42": 0.52918},
                   "verdict": "PASS" if permuted_delta < 0.01 and
                              report["runs"]["permuted_champion"]["oof_macro_f1"] < chance * 3
                              else "FAIL"})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 66)
    print(f"{'':26s}{'B04':>13s}{'shrink10':>13s}{'delta':>13s}")
    print("-" * 66)
    print(f"{'실제 label':24s}{report['runs']['real_b04']['oof_macro_f1']:>13.6f}"
          f"{report['runs']['real_champion']['oof_macro_f1']:>13.6f}{real_delta:>+13.6f}")
    print(f"{'섞인 label':24s}{report['runs']['permuted_b04']['oof_macro_f1']:>13.6f}"
          f"{report['runs']['permuted_champion']['oof_macro_f1']:>13.6f}{permuted_delta:>+13.6f}")
    print("-" * 66)
    print(f"SDH 보고 (seed42)         {0.47786:>13.5f}{0.52918:>13.5f}{0.52918-0.47786:>+13.5f}")
    print(f"우연 수준 (1/{report['n_classes']})              {chance:.6f}")
    print("=" * 66)
    if report["verdict"] == "PASS":
        print("✅ PASS — 섞인 label 에서 enrichment 가 B04 를 넘지 못한다.")
        print("   SDH 실제 코드 기준이므로 exp_009 감사의 '문서만 검증' 한계가 닫혔다.")
    else:
        print("🚨 FAIL — 섞인 label 에서 enrichment 가 B04 를 넘었다. 제출하면 안 된다.")
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
