"""
제출 파일 생성 — GBV C=0.1  및  GBV + freq 토큰 100
=====================================================

    .venv/bin/python3 experiments/member_d/exp_003_discriminative_tokens/submit.py

두 개를 만든다:
  1) submission_GBV_C0.1_cv0.41202.csv           토큰 없음 (LR-002 앵커와 비교용)
  2) submission_GBV+freqtok100_C0.1_cv0.41838.csv  빈도 토큰 100 (CV 최고)

----------------------------------------------------------------------
왜 이렇게 두 장인가
----------------------------------------------------------------------
세션 내내 CV 를 올렸지만 LB 는 한 번도 확인하지 않았다. 유일한 실측은
LR-002 (GBVR C=1.0) CV 0.384 → LB 0.260 으로 간격이 ~0.12 다.

  · GBV C=0.1  은 LR-002 와 거의 같은 파이프라인(C·R블록만 다름)이라
    "CV 개선이 LB 로 가는가"를 가장 깨끗하게 잰다.
  · GBV+토큰100 은 그 위에 토큰 FE 가 LB 로 가는가를 잰다.

두 점이면 CV→LB 기울기가 보인다. 간격이 일정하면 CV 를 믿고 최적화를 계속,
벌어지면 우리가 train 배치 구조에 과적합 중이라는 뜻이다.

----------------------------------------------------------------------
규칙 준수
----------------------------------------------------------------------
· 토큰 선택은 **전체 train 라벨로만** 한다 (test 는 존재 여부만 적용). 규칙 2.
· 외부 유전자-암종 지식은 쓰지 않는다. 규칙 1.
· test 는 예측에만 쓰고 통계를 전처리에 넣지 않는다.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

_HERE = Path(__file__).resolve().parent
_EXP2 = _HERE.parent / "exp_002_variant_type"
for p in (_HERE, _EXP2):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pipeline as pa                                                # noqa: E402
import features_A as fa                                              # noqa: E402
import features_D as fd                                              # noqa: E402
from run_tokens import load_cfg                                      # noqa: E402


def build_full(train, test, counts_tr, counts_te, gene_cols, blocks,
               tok_train=None, tok_test=None, y=None, spec_tok=None, k=0):
    """전체 train 으로 spec 을 fit 하고 train·test 피처를 만든다."""
    spec = fa.fit_spec(train, gene_cols, seed=42)
    Xtr, _ = fa.build_features(train, counts_tr, spec, blocks)
    Xte, _ = fa.build_features(test, counts_te, spec, blocks)
    if k > 0:
        idx_all = np.arange(len(train))
        Dtr, _ = fd.transform_tokens(tok_train, idx_all, spec_tok, k)
        idx_te = np.arange(len(test))
        Dte, _ = fd.transform_tokens(tok_test, idx_te, spec_tok, k)
        Xtr = sparse.hstack([Xtr, Dtr], format="csr")
        Xte = sparse.hstack([Xte, Dte], format="csr")
    return Xtr, Xte


def cv_seed42(train, y, counts, gene_cols, blocks, mp, fn, model_seed,
              n_splits, tok_sets=None, spec_maker=None, k=0):
    """seed 42 5-fold OOF Macro F1 — 제출 모델이 기록된 CV 와 맞는지 확인."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.empty(len(y), dtype=object)
    for i_tr, i_va in cv.split(train, y):
        spec = fa.fit_spec(train.iloc[i_tr], gene_cols, seed=model_seed)
        Xa, _ = fa.build_features(train.iloc[i_tr], counts.iloc[i_tr], spec, blocks)
        Xb, _ = fa.build_features(train.iloc[i_va], counts.iloc[i_va], spec, blocks)
        if k > 0:
            st = spec_maker(i_tr)
            Da, _ = fd.transform_tokens(tok_sets, i_tr, st, k)
            Db, _ = fd.transform_tokens(tok_sets, i_va, st, k)
            Xa = sparse.hstack([Xa, Da], format="csr")
            Xb = sparse.hstack([Xb, Db], format="csr")
        oof[i_va] = fn(model_seed, mp).fit(Xa, y[i_tr]).predict(Xb)
    oof = np.array(list(oof))
    return round(float(f1_score(y, oof, average="macro")), 5), round(
        float(accuracy_score(y, oof)), 5)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="배선 점검 (점수 무의미)")
    a = ap.parse_args()

    root = pa.find_project_root()
    cfg = load_cfg()
    blocks = tuple(cfg["blocks"])
    mp = dict(cfg["model_params"])
    model_seed = cfg.get("model_seed", 42)
    n_splits = 2 if a.smoke else cfg["n_splits"]
    min_count = cfg["min_count"]
    name, fn = pa.MODELS[cfg["model"]]

    out = root / "experiments" / "member_d" / "results" / "member-d-exp003"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  제출 파일 생성 — GBV C=0.1  및  GBV + freq 토큰 100")
    print(f"  {mp}")
    print("=" * 78)

    train, test, sample_sub, gene_cols = pa.load_data(root, smoke=a.smoke)
    y = train[pa.TARGET].values
    counts_tr, counts_te = pa.parse_all(train, test, gene_cols)

    print("토큰 파싱 (train·test)...", flush=True)
    tok_tr = fd.parse_token_sets(train, gene_cols)
    tok_te = fd.parse_token_sets(test, gene_cols)

    ids = test[pa.ID].values

    def write_sub(pred, tag, cv):
        df = pd.DataFrame({pa.ID: ids, pa.TARGET: pred})
        assert len(df) == len(test), "행 수 불일치"
        assert df[pa.TARGET].notna().all(), "NaN 예측 있음"
        unknown = set(df[pa.TARGET]) - set(y)
        assert not unknown, f"train 에 없는 라벨 예측: {unknown}"
        f = out / f"submission_{tag}_cv{cv:.5f}.csv"
        df.to_csv(f, index=False)
        print(f"  저장: {f.name}  (분포 상위: "
              f"{df[pa.TARGET].value_counts().head(3).to_dict()})")
        return f

    # ── 1) GBV C=0.1 (토큰 없음) ──────────────────────────────────────
    print("\n[1] GBV C=0.1 (토큰 없음)")
    t0 = time.time()
    cv1, acc1 = cv_seed42(train, y, counts_tr, gene_cols, blocks, mp, fn,
                          model_seed, n_splits, k=0)
    print(f"  seed42 CV {cv1:.5f} / Acc {acc1:.5f}  (기록값 0.41482 와 대조)")
    Xtr, Xte = build_full(train, test, counts_tr, counts_te, gene_cols, blocks)
    pred1 = fn(model_seed, mp).fit(Xtr, y).predict(Xte)
    write_sub(pred1, "GBV_C0.1", cv1)
    print(f"  ({time.time() - t0:.0f}s)")

    # ── 2) GBV + freq 토큰 100 ────────────────────────────────────────
    print("\n[2] GBV + freq 토큰 100")
    t0 = time.time()

    def spec_maker(idx_tr):
        return fd.fit_tokens(tok_tr, y, idx_tr, top_k=100,
                             min_count=min_count, method="freq")

    cv2, acc2 = cv_seed42(train, y, counts_tr, gene_cols, blocks, mp, fn,
                          model_seed, n_splits, tok_sets=tok_tr,
                          spec_maker=spec_maker, k=100)
    print(f"  seed42 CV {cv2:.5f} / Acc {acc2:.5f}  (기록값 0.42104 와 대조)")
    spec_full = fd.fit_tokens(tok_tr, y, np.arange(len(train)), top_k=100,
                              min_count=min_count, method="freq")
    print(f"  최종 토큰 후보 {spec_full['n_candidates']}개 중 100 선택 · "
          f"아티팩트 플래그 {spec_full['diag']['n_flagged_artifact']}개")
    Xtr, Xte = build_full(train, test, counts_tr, counts_te, gene_cols, blocks,
                          tok_train=tok_tr, tok_test=tok_te, y=y,
                          spec_tok=spec_full, k=100)
    pred2 = fn(model_seed, mp).fit(Xtr, y).predict(Xte)
    write_sub(pred2, "GBV+freqtok100_C0.1", cv2)
    print(f"  ({time.time() - t0:.0f}s)")

    print("\n" + "=" * 78)
    print("  두 파일 다 results/member-d-exp003/ 에 있습니다.")
    print("  데이콘에 업로드 후 LB 점수를 실험·제출 기록 DB 에 적어주세요.")
    print("  두 CV(0.412 / 0.418)가 LB 에서 어떻게 벌어지는지가 핵심입니다.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
