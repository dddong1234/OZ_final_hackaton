"""
클래스별 진단 — 어디가 왜 막혀 있는가
======================================

    .venv/bin/python3 experiments/iljun/exp_002_variant_type/diagnose_classes.py

----------------------------------------------------------------------
왜 만드나
----------------------------------------------------------------------
지금까지 우리는 **Macro F1 평균 하나**만 보고 판단했다. 그런데 평균은 속인다.
`V`(변이 유형) 블록을 넣었을 때 16개 클래스가 오르고 **10개가 내렸는데**,
평균만 봐서는 그게 안 보였다.

그리고 F1 이 낮은 데는 **성격이 다른 두 가지 이유**가 있다.

    A형 · 과잉 예측   정밀도 낮음 / 재현율 높음
                      → 이 암종이 아닌 사람까지 이 암종이라고 찍는다
                      → 처방: 덜 찍게 (class_weight·임계값 조정)

    B형 · 누락        정밀도 높음 / 재현율 낮음
                      → 찍은 건 맞는데 대부분 못 찾는다
                      → 처방: 신호를 더 주기 (피처 추가)

**처방이 정반대다.** 어느 쪽인지 모르고 손대면 헛수고다.
`class_weight="balanced"` 는 소수 클래스를 더 많이 찍게 만드는 설정이므로
지금 낮은 클래스들이 A형일 가능성이 있는데, 확인한 적이 없다.

----------------------------------------------------------------------
무엇을 뽑나
----------------------------------------------------------------------
1. 클래스별 정밀도 · 재현율 · F1  (cv_seeds 3개 평균 ± 표준편차)
2. A형 / B형 진단
3. 혼동 쌍 — 어떤 암종을 어떤 암종으로 착각하는가
4. 무리별 요약 (A 지킬 것 / B 막힌 것 / C 여지 있는 것)

train 만 사용한다. test 는 열지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pipeline as pa                                                # noqa: E402

# 2026-07-31 분석에서 나온 무리 구분
GROUP = {
    "A 지킬 것":   ["ACC", "SKCM", "THCA", "COAD", "LAML"],
    "B 막힌 것":   ["KIRC", "KIPAN", "LGG", "GBMLGG"],
    "C 여지 있음": ["CESC", "PAAD", "LUAD", "BLCA", "LIHC", "HNSC", "SARC", "THYM", "OV"],
}
# 마커 0개 클래스 (gene_class_table.py 결과)
NO_MARKER = ["BRCA", "OV", "SARC", "THYM"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="클래스별 정밀도·재현율 진단")
    ap.add_argument("--root", default=None)
    ap.add_argument("--blocks", default=None, help="기본은 config.yaml 의 blocks")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    root = Path(a.root) if a.root else pa.find_project_root()
    cfg = pa.load_cfg()["pipeline"]
    blocks = a.blocks or cfg["blocks"]
    cv_seeds = tuple(cfg["cv"]["seeds"])[: (2 if a.smoke else None)]
    model_seed = cfg["cv"].get("model_seed", pa.MODEL_SEED)
    n_splits = 2 if a.smoke else cfg["cv"]["n_splits"]
    mp = dict(cfg.get("model_params") or pa.DEFAULT_MODEL_PARAMS[cfg["model"]])

    print("=" * 84)
    print("  클래스별 진단 — 정밀도 · 재현율 · 혼동")
    print(f"  피처 {blocks} · {mp} · cv_seeds {list(cv_seeds)} · KFold-{n_splits}")
    print("=" * 84)

    train, test, _, gene_cols = pa.load_data(root, smoke=a.smoke)
    y = train[pa.TARGET].values
    cnt_train, _ = pa.parse_all(train, test, gene_cols)
    classes = sorted(pd.unique(y))

    # ── seed 별로 OOF 를 받아 클래스별 지표 계산 ─────────────────────
    per_seed_rows, oof_by_seed = [], {}
    for s in cv_seeds:
        r = pa.cross_validate(train, y, cnt_train, gene_cols, tuple(blocks),
                              model_key=cfg["model"], model_params=mp,
                              cv_seed=s, model_seed=model_seed,
                              n_splits=n_splits, v=False)
        oof_by_seed[s] = r["oof"]
        p, rc, f, sup = precision_recall_fscore_support(
            y, r["oof"], labels=classes, zero_division=0)
        for i, c in enumerate(classes):
            per_seed_rows.append({"암종": c, "cv_seed": s, "정밀도": p[i],
                                  "재현율": rc[i], "F1": f[i], "환자수": int(sup[i])})
        print(f"  cv_seed {s}  Macro F1 {r['f1_macro']:.5f}")

    df = pd.DataFrame(per_seed_rows)
    agg = df.groupby("암종").agg(
        환자수=("환자수", "first"),
        정밀도=("정밀도", "mean"), 정밀도σ=("정밀도", lambda x: x.std(ddof=1)),
        재현율=("재현율", "mean"), 재현율σ=("재현율", lambda x: x.std(ddof=1)),
        F1=("F1", "mean"), F1σ=("F1", lambda x: x.std(ddof=1)),
    ).round(4)

    # ── A형 / B형 진단 ───────────────────────────────────────────────
    agg["재현−정밀"] = (agg["재현율"] - agg["정밀도"]).round(4)

    def diagnose(row):
        d = row["재현−정밀"]
        if row["F1"] >= 0.50:
            return "양호"
        if d > 0.10:
            return "A 과잉예측"          # 많이 찍는데 틀린 게 많다
        if d < -0.10:
            return "B 누락"              # 정확한데 못 찾는다
        return "— 균형(신호부족)"

    agg["진단"] = agg.apply(diagnose, axis=1)
    agg["마커"] = ["없음" if c in NO_MARKER else "" for c in agg.index]
    agg = agg.sort_values("F1")

    print("\n" + "=" * 84)
    print("  클래스별 정밀도 · 재현율 (F1 오름차순)")
    print("=" * 84)
    show = agg[["환자수", "정밀도", "재현율", "F1", "F1σ", "재현−정밀", "진단", "마커"]]
    print(show.to_string())

    print("\n── 진단별 개수 " + "─" * 60)
    print(agg["진단"].value_counts().to_string())
    print("\n  A 과잉예측 = 덜 찍게 만들면 개선 (class_weight·임계값)")
    print("  B 누락     = 신호를 더 줘야 개선 (피처 추가)")
    print("  균형(신호부족) = 정밀도·재현율 둘 다 낮음. 구분 재료 자체가 부족")

    # ── 무리별 요약 ──────────────────────────────────────────────────
    print("\n── 무리별 요약 " + "─" * 60)
    rows = []
    assigned = set()
    for name, members in GROUP.items():
        m = [c for c in members if c in agg.index]
        assigned |= set(m)
        rows.append({"무리": name, "클래스 수": len(m),
                     "평균 F1": round(agg.loc[m, "F1"].mean(), 4),
                     "평균 정밀도": round(agg.loc[m, "정밀도"].mean(), 4),
                     "평균 재현율": round(agg.loc[m, "재현율"].mean(), 4),
                     "지분": f"{len(m)}/26 = {len(m)/26:.1%}"})
    rest = [c for c in agg.index if c not in assigned]
    rows.append({"무리": "나머지", "클래스 수": len(rest),
                 "평균 F1": round(agg.loc[rest, "F1"].mean(), 4),
                 "평균 정밀도": round(agg.loc[rest, "정밀도"].mean(), 4),
                 "평균 재현율": round(agg.loc[rest, "재현율"].mean(), 4),
                 "지분": f"{len(rest)}/26 = {len(rest)/26:.1%}"})
    gtab = pd.DataFrame(rows)
    print(gtab.to_string(index=False))

    # ── 혼동 쌍 (첫 seed 기준) ───────────────────────────────────────
    s0 = cv_seeds[0]
    cm = confusion_matrix(y, oof_by_seed[s0], labels=classes)
    pairs = []
    for i, ci in enumerate(classes):
        n_i = cm[i].sum()
        for j, cj in enumerate(classes):
            if i != j and cm[i, j] > 0:
                pairs.append({"실제": ci, "예측": cj, "건수": int(cm[i, j]),
                              "그 암종의 %": round(cm[i, j] / n_i * 100, 1)})
    pdf = pd.DataFrame(pairs).sort_values("건수", ascending=False)
    print(f"\n── 혼동 쌍 상위 15 (cv_seed {s0}) " + "─" * 45)
    print(pdf.head(15).to_string(index=False))

    # 상호 혼동 (양방향 모두 큰 쌍)
    print("\n── 서로 헷갈리는 쌍 (양방향 합계 상위 8) " + "─" * 33)
    mutual = {}
    for r in pairs:
        k = tuple(sorted([r["실제"], r["예측"]]))
        mutual[k] = mutual.get(k, 0) + r["건수"]
    mut = pd.DataFrame([{"쌍": f"{a} ↔ {b}", "양방향 합계": v}
                        for (a, b), v in mutual.items()]).sort_values("양방향 합계", ascending=False)
    print(mut.head(8).to_string(index=False))

    if not a.smoke:
        art = root / "experiments" / "iljun" / "exp_002_variant_type" / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        agg.to_csv(art / "class_diagnosis.csv", encoding="utf-8-sig")
        pdf.to_csv(art / "confusion_pairs.csv", index=False, encoding="utf-8-sig")
        gtab.to_csv(art / "group_summary.csv", index=False, encoding="utf-8-sig")
        (art / "class_diagnosis.json").write_text(json.dumps({
            "blocks": blocks, "model_params": mp,
            "cv_seeds": [int(s) for s in cv_seeds], "n_splits": int(n_splits),
            "per_class": agg.reset_index().to_dict("records"),
            "groups": rows,
            "top_mutual_pairs": mut.head(8).to_dict("records"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n저장: artifacts/class_diagnosis.csv · confusion_pairs.csv · "
              "group_summary.csv · class_diagnosis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
