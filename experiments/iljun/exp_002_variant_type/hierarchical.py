"""
계층형 분류 — 묶을 쌍을 fold 학습 분할에서 스스로 찾는다
========================================================

    .venv/bin/python3 experiments/iljun/exp_002_variant_type/hierarchical.py
    .venv/bin/python3 experiments/iljun/exp_002_variant_type/hierarchical.py --smoke

----------------------------------------------------------------------
★ 대회 규칙 (2026-07-31 주최측 확인) 과 이 코드의 관계
----------------------------------------------------------------------
1. 외부 논문의 유전자-암종 연관성 → 모델 입력·피처 선택·임계값에 **전혀 사용 불가**.
   발표에서 사후 생물학적 해석·참고문헌으로만 인용 가능.
2. train 내부에서 계산한 유전자-암종 연관 후보를 피처로 쓰는 것 → **허용**.
   단 각 fold 의 **학습 분할에서만 산출**하고 validation 에는 적용만 한다.
3. KIPAN/KIRC, GBMLGG/LGG 같은 공개된 코호트 명칭 관계를 **모델 구조나 규칙에
   반영하지 않고** 발표에서 오류 분석 가설로만 언급하는 것 → 허용.

**3번이 허가한 것은 "반영하지 않는" 쪽이다.** 반영해도 된다는 뜻이 아니다.
그래서 이 코드는 KIPAN·KIRC 같은 **라벨 이름을 일절 쓰지 않는다.**

대신 2번이 허용한 방식으로 간다 — **묶을 쌍을 fold 학습 분할에서 계산해 찾는다.**
찾아낸 쌍이 결과적으로 무엇이든, 그건 데이터가 말한 것이지 내가 넣은 것이 아니다.
그 쌍이 왜 나왔는지에 대한 TCGA 코호트 설명은 **발표에서 사후 해석으로만** 쓴다.

----------------------------------------------------------------------
왜 계층형인가 — tune_class_bias.py 의 실패에서 나온 결론
----------------------------------------------------------------------
확률에 배수를 곱해 예측을 바꿔봤더니 **완전한 제로섬**이었다.

    조정한 클래스 합계  +0.0699
    나머지    합계      -0.0659      → 거의 상쇄

argmax 는 26개 중 **가장 큰 하나**를 고르는 구조라, 한 클래스를 더 찍게 하면
반드시 다른 클래스가 덜 찍힌다. 파이 크기는 그대로고 조각만 옮겨진다.
파이를 키우려면 **경쟁 구도 자체를 바꿔야 한다.** 그게 계층형이다.

----------------------------------------------------------------------
어떻게 쌍을 찾나 — fold 학습 분할의 클래스 중심 유사도
----------------------------------------------------------------------
두 클래스가 사실상 같은 집단이면, 그 두 집단의 **평균 피처 벡터(중심)** 가
거의 같은 방향을 가리킨다. 그래서 fold 학습 분할에서만 이렇게 한다.

    1. 클래스마다 학습 분할 환자들의 피처 평균을 낸다  (= 중심)
    2. 중심끼리 코사인 유사도를 잰다
    3. 가장 닮은 쌍 top-k 를 고른다 (한 클래스는 한 쌍에만)
    4. 그 쌍을 1단계에서 하나로 묶는다

validation 분할은 **여기에 전혀 관여하지 않는다.** 고른 쌍을 적용만 받는다.
fold 마다 다시 계산하므로 fold 밖 정보가 새지 않는다.

----------------------------------------------------------------------
2단계 구조
----------------------------------------------------------------------
    [1단계]  묶은 쌍을 한 덩어리로 보고 26-k 개 클래스로 분류
    [2단계]  그 덩어리로 예측된 환자만 다시 원래 두 클래스로 나눈다
             (2단계 분류기도 fold 학습 분할의 해당 클래스 환자만으로 학습)

**왜 도움이 될 수 있나.** 같은 집단인 두 클래스는 확률이 반씩 쪼개진다. 그렇게
나뉜 확률이 엉뚱한 클래스에게 지기도 한다. 합치면 확률이 한곳에 모인다.

**왜 안 될 수도 있나.** 2단계가 동전 던지기면 합쳐서 번 것을 그대로 잃는다.
Macro F1 은 26개 평균이라 **두 클래스 모두** 점수가 나와야 한다.

----------------------------------------------------------------------
그래서 순서대로 재본다 (섣불리 결론 내지 않기 위해)
----------------------------------------------------------------------
[1] 어떤 쌍이 찾아졌나  — fold 마다 같은 쌍이 나오나 (안정성)
[2] 1단계 단독          — 묶었을 때 점수
[3] 2단계 단독          — 그룹 안에서 나누는 게 가능하기는 한가
[4] 오라클 상한         — 2단계가 완벽하다면 최대 얼마인가
                          ★ 이 값이 낮으면 이 방향은 시작할 가치가 없다
[5] 실제 결합           — 평면 모델과 같은 fold 로 paired 비교
[6] 판정                — 3 seed 방향 일관 + 크기가 σ 를 넘어야 '검출'

폴드 분할은 원래 26개 라벨로 stratify 한다. 그래야 평면 모델과 완전히 같은
폴드가 되어 숫자를 직접 뺄 수 있다 (paired 비교).

train 만 사용한다. test 는 열지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pipeline as pa                                                # noqa: E402
import features_A as fa                                              # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# 쌍 찾기 — fold 학습 분할에서만
# ──────────────────────────────────────────────────────────────────────
def class_centroids(X, y_tr, classes):
    """클래스마다 학습 분할 환자들의 피처 평균 벡터를 만든다."""
    return np.vstack([np.asarray(X[np.where(y_tr == c)[0]].mean(axis=0)).ravel()
                      for c in classes])


def discover_pairs(X, y_tr, top_k=2, min_sim=0.0):
    """중심끼리 가장 닮은 쌍 top_k 를 고른다. 한 클래스는 한 쌍에만 들어간다.

    학습 분할 X·y_tr 만 본다. validation 은 관여하지 않는다.

    ★ 중심을 그대로 비교하면 안 된다. TP53 처럼 거의 모든 암종에 흔한 유전자가
      모든 중심에 공통으로 들어 있어서, 어느 쌍을 봐도 다 닮아 보인다.
      그래서 **전체 평균 중심을 빼고** 남은 '이 클래스만의 편차'끼리 비교한다.
      같은 집단에서 나온 두 클래스는 편차 방향까지 같다.
    """
    classes = np.array(sorted(set(y_tr)))
    M = class_centroids(X, y_tr, classes)
    M = M - M.mean(axis=0, keepdims=True)          # 공통 성분 제거
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    S = (M / norms) @ (M / norms).T
    np.fill_diagonal(S, -np.inf)

    used, pairs = set(), []
    order = np.dstack(np.unravel_index(np.argsort(-S, axis=None), S.shape))[0]
    for i, j in order:
        if len(pairs) >= top_k:
            break
        if i >= j:                                   # 위쪽 삼각만
            continue
        a, b = classes[i], classes[j]
        if a in used or b in used or S[i, j] < min_sim:
            continue
        used |= {a, b}
        pairs.append((str(a), str(b), float(S[i, j])))
    return pairs


def build_mapping(pairs):
    """찾은 쌍을 {클래스: 그룹이름} 으로. 그룹 이름은 라벨 정보를 담지 않는다."""
    m = {}
    for k, (a, b, _) in enumerate(pairs, 1):
        m[a] = f"__G{k}__"
        m[b] = f"__G{k}__"
    members = {f"__G{k}__": [a, b] for k, (a, b, _) in enumerate(pairs, 1)}
    return m, members


def to_group(y, mapping):
    return np.array([mapping.get(v, v) for v in y], dtype=object)


def macro_f1(y, pred):
    return float(f1_score(y, pred, average="macro"))


def paired(after: dict, before: dict):
    """같은 cv_seed 끼리 짝지어 뺀다. 증분의 σ 가 작아져 민감해진다."""
    seeds = sorted(set(after) & set(before))
    d = np.array([after[s] - before[s] for s in seeds], dtype=float)
    sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    return float(d.mean()), sd, int((d > 0).sum()), len(d)


# ──────────────────────────────────────────────────────────────────────
def run_one_seed(train, y, counts, gene_cols, blocks, model_key, mp,
                 cv_seed, model_seed, n_splits, top_k, v=True):
    """cv_seed 하나에 대해 평면·계층·오라클을 **같은 폴드에서** 한 번에 계산한다."""
    name, fn = pa.MODELS[model_key]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)

    n = len(y)
    oof_flat = np.empty(n, dtype=object)
    oof_hier = np.empty(n, dtype=object)
    oof_orac = np.empty(n, dtype=object)
    stage1_true, stage1_pred = np.empty(n, dtype=object), np.empty(n, dtype=object)
    stage2_rows, fold_pairs = [], []

    t0 = time.time()
    for k, (i_tr, i_va) in enumerate(cv.split(train, y), 1):
        spec = fa.fit_spec(train.iloc[i_tr], gene_cols, seed=model_seed)
        Xa, _ = fa.build_features(train.iloc[i_tr], counts.iloc[i_tr], spec, blocks)
        Xb, _ = fa.build_features(train.iloc[i_va], counts.iloc[i_va], spec, blocks)
        y_tr, y_va = y[i_tr], y[i_va]

        # ── 대조군: 평면 26-way ──────────────────────────────────────
        oof_flat[i_va] = fn(model_seed, mp).fit(Xa, y_tr).predict(Xb)

        # ── ★ 묶을 쌍을 학습 분할에서만 찾는다 ───────────────────────
        pairs = discover_pairs(Xa, y_tr, top_k=top_k)
        mapping, members = build_mapping(pairs)
        fold_pairs.append({"cv_seed": int(cv_seed), "fold": k,
                           "pairs": [(a, b, round(s, 4)) for a, b, s in pairs]})

        # ── 1단계 ────────────────────────────────────────────────────
        g_tr = to_group(y_tr, mapping)
        g_pred = fn(model_seed, mp).fit(Xa, g_tr).predict(Xb)
        stage1_true[i_va] = to_group(y_va, mapping)
        stage1_pred[i_va] = g_pred

        pred_h = np.array(g_pred, dtype=object)
        pred_o = np.array(g_pred, dtype=object)

        # ── 2단계: 그룹마다 별도 분류기 (학습 분할의 해당 환자만) ────
        for gname, ms in members.items():
            idx_tr = np.where(np.isin(y_tr, ms))[0]
            sel = np.where(g_pred == gname)[0]
            if len(idx_tr) < 2 or len(np.unique(y_tr[idx_tr])) < 2:
                pred_h[sel] = ms[0]
                pred_o[sel] = ms[0]
                continue

            clf2 = fn(model_seed, mp).fit(Xa[idx_tr], y_tr[idx_tr])
            if len(sel):
                pred_h[sel] = clf2.predict(Xb[sel])
                # 오라클: 2단계가 완벽하다고 가정
                in_g = np.isin(y_va[sel], ms)
                pred_o[sel] = np.where(in_g, y_va[sel], ms[0])

            # 2단계 단독: **실제** 그 쌍에 속한 환자만으로 정확도를 잰다
            idx_va_true = np.where(np.isin(y_va, ms))[0]
            if len(idx_va_true):
                p2 = clf2.predict(Xb[idx_va_true])
                vc = pd.Series(y_va[idx_va_true]).value_counts(normalize=True)
                stage2_rows.append({
                    "쌍": " + ".join(ms), "cv_seed": int(cv_seed), "fold": k,
                    "검증 환자수": len(idx_va_true),
                    "정확도": float(accuracy_score(y_va[idx_va_true], p2)),
                    "다수클래스 비율": float(vc.iloc[0]),
                    "Macro F1": float(f1_score(y_va[idx_va_true], p2,
                                               average="macro", zero_division=0)),
                })

        oof_hier[i_va] = pred_h
        oof_orac[i_va] = pred_o

    out = {
        "cv_seed": int(cv_seed),
        "flat": round(macro_f1(y, np.array(list(oof_flat))), 5),
        "hier": round(macro_f1(y, np.array(list(oof_hier))), 5),
        "oracle": round(macro_f1(y, np.array(list(oof_orac))), 5),
        "flat_acc": round(float(accuracy_score(y, np.array(list(oof_flat)))), 5),
        "hier_acc": round(float(accuracy_score(y, np.array(list(oof_hier)))), 5),
        "stage1_f1": round(macro_f1(np.array(list(stage1_true)),
                                    np.array(list(stage1_pred))), 5),
        "oof_flat": np.array(list(oof_flat)),
        "oof_hier": np.array(list(oof_hier)),
        "stage2": stage2_rows,
        "fold_pairs": fold_pairs,
        "secs": round(time.time() - t0),
    }
    if v:
        print(f"  cv_seed {cv_seed}  평면 {out['flat']:.5f}  계층 {out['hier']:.5f}  "
              f"오라클 {out['oracle']:.5f}   1단계 {out['stage1_f1']:.5f}"
              f"   ({out['secs']}s)", flush=True)
    return out


# ──────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="계층형 분류 — 묶을 쌍을 fold 학습 분할에서 찾는다")
    ap.add_argument("--root", default=None)
    ap.add_argument("--blocks", default=None, help="기본은 config.yaml 의 blocks")
    ap.add_argument("--top-k", type=int, default=2, help="묶을 쌍의 개수 (기본 2)")
    ap.add_argument("--smoke", action="store_true", help="2 seed × 2 fold 배선 점검")
    a = ap.parse_args(argv)

    root = Path(a.root) if a.root else pa.find_project_root()
    cfg = pa.load_cfg()["pipeline"]
    blocks = tuple(a.blocks or cfg["blocks"])
    cv_seeds = tuple(cfg["cv"]["seeds"])[: (2 if a.smoke else None)]
    model_seed = cfg["cv"].get("model_seed", pa.MODEL_SEED)
    n_splits = 2 if a.smoke else cfg["cv"]["n_splits"]
    model_key = cfg["model"]
    mp = dict(cfg.get("model_params") or pa.DEFAULT_MODEL_PARAMS[model_key])

    print("=" * 88)
    print("  계층형 분류 — 묶을 쌍을 fold 학습 분할에서 찾는다")
    print(f"  피처 {''.join(blocks)} · {pa.MODELS[model_key][0]} · {mp}")
    print(f"  cv_seeds {list(cv_seeds)} · model_seed {model_seed} · "
          f"StratifiedKFold-{n_splits} · top_k {a.top_k}")
    print("  ※ 라벨 이름을 코드에 쓰지 않는다 (대회 규칙 3번). 쌍은 데이터가 고른다.")
    print("=" * 88)

    train, test, _, gene_cols = pa.load_data(root, smoke=a.smoke)
    y = train[pa.TARGET].values
    cnt_train, _ = pa.parse_all(train, test, gene_cols)

    res = [run_one_seed(train, y, cnt_train, gene_cols, blocks, model_key, mp,
                        s, model_seed, n_splits, a.top_k) for s in cv_seeds]

    flat = {r["cv_seed"]: r["flat"] for r in res}
    hier = {r["cv_seed"]: r["hier"] for r in res}
    orac = {r["cv_seed"]: r["oracle"] for r in res}

    def stat(d):
        v = np.array(list(d.values()), dtype=float)
        return v.mean(), (v.std(ddof=1) if len(v) > 1 else float("nan"))

    mf, sf = stat(flat); mh, sh = stat(hier); mo, so = stat(orac)

    # ── [1] 어떤 쌍이 찾아졌나 ────────────────────────────────────────
    print("\n[1] 어떤 쌍이 찾아졌나 — fold 마다 다시 계산한 결과")
    allp = [fp for r in res for fp in r["fold_pairs"]]
    cnt = Counter(tuple(sorted((a_, b_))) for fp in allp for a_, b_, _ in fp["pairs"])
    n_folds = len(allp)
    prow = [{"쌍": f"{x} + {z}", "찾아진 fold": f"{c}/{n_folds}",
             "비율": f"{c / n_folds:.0%}"} for (x, z), c in cnt.most_common()]
    print(pd.DataFrame(prow).to_string(index=False))
    print(f"\n  전체 {n_folds}개 fold 에서 각각 top-{a.top_k} 쌍을 다시 골랐다.")
    print("  같은 쌍이 매번 나오면 데이터에 실재하는 구조이고,")
    print("  fold 마다 달라지면 우연히 닮아 보인 것이다.")

    # ── [3] 2단계 단독 ────────────────────────────────────────────────
    s2 = pd.DataFrame([r for x in res for r in x["stage2"]])
    print("\n[3] 2단계 단독 — 그룹 안에서 나누는 게 가능하기는 한가")
    agg2 = pd.DataFrame()
    if len(s2):
        agg2 = s2.groupby("쌍").agg(측정횟수=("정확도", "size"),
                                    정확도=("정확도", "mean"),
                                    다수클래스=("다수클래스 비율", "mean"),
                                    MacroF1=("Macro F1", "mean")).round(4)
        agg2["다수클래스 대비"] = (agg2["정확도"] - agg2["다수클래스"]).round(4)
        print(agg2.sort_values("다수클래스 대비", ascending=False).to_string())
        print("\n  '다수클래스' 는 무조건 많은 쪽으로만 찍었을 때의 정확도다.")
        print("  '다수클래스 대비' 가 0 근처면 2단계는 동전 던지기이고 계층형은 의미가 없다.")
    else:
        print("  (2단계 표본 없음)")

    # ── [4] 오라클 상한 ───────────────────────────────────────────────
    print("\n[4] 오라클 상한 — 2단계가 완벽하다면")
    print(f"  평면    {mf:.5f} ± {sf:.5f}")
    print(f"  오라클  {mo:.5f} ± {so:.5f}   (차이 {mo - mf:+.5f})")
    print("  → 이 차이가 이 구조로 벌 수 있는 **최대치**다. 실제는 반드시 이보다 낮다.")

    # ── [5] 실제 결합 ─────────────────────────────────────────────────
    print("\n[5] 실제 결합 결과")
    tab = pd.DataFrame([{"cv_seed": r["cv_seed"], "평면": r["flat"], "계층형": r["hier"],
                         "차이": round(r["hier"] - r["flat"], 5),
                         "오라클": r["oracle"], "1단계": r["stage1_f1"]} for r in res])
    print(tab.to_string(index=False))
    m, sd, pos, nn = paired(hier, flat)
    print(f"\n  계층형 {mh:.5f} ± {sh:.5f}   vs   평면 {mf:.5f} ± {sf:.5f}")
    print(f"  paired 증분 {m:+.5f} ± {sd:.5f}   ({pos}/{nn} seed 에서 양수)")

    # ── [6] 판정 ──────────────────────────────────────────────────────
    all_up = (pos == nn) and nn > 1
    all_down = (pos == 0) and nn > 1
    big = abs(m) >= (sd if np.isfinite(sd) and sd > 0 else 0)
    detected = all_up and big
    print("\n[6] 판정")
    if all_up and big:
        print("  → 방향이 일관되고 크기도 증분 σ 를 넘는다. 실제 효과로 볼 근거가 있다.")
    elif all_up:
        print("  → 방향은 일관되나 크기가 증분 σ 안에 있다. 효과가 있다고 말할 수 없다.")
    elif all_down and big:
        print(f"  → **{nn}개 seed 전부에서 나빠졌다** (평균 {m:+.5f}, σ 초과).")
        print("     우연이 아니라 이 구조가 실제로 손해라는 뜻이다. 기각한다.")
    elif all_down:
        print(f"  → {nn}개 seed 전부 음수지만 크기가 증분 σ 안이다. 이득은 없다고 본다.")
    else:
        print("  → seed 에 따라 방향이 갈린다. 개선으로 볼 수 없다.")
    print("  ※ n=3 의 σ 는 그 자체로 부정확하다. 방향과 대략적 크기만 읽는다.")

    # 2단계가 진짜 범인인지 — 오라클이 크면 구조가 아니라 2단계가 문제다
    if mo - mf > 0 and mh < mf:
        print(f"\n  ★ 오라클은 평면보다 {mo - mf:+.5f} 높은데 실제는 {m:+.5f} 다.")
        print("     1단계(묶기)는 되는데 2단계(다시 나누기)에서 전부 잃는다는 뜻이다.")
        print("     구조가 틀린 게 아니라 2단계를 풀 재료가 없는 것이다.")

    # ── [7] 클래스별 변화 ─────────────────────────────────────────────
    classes = sorted(pd.unique(y))
    touched = {c for (x, z) in cnt for c in (x, z)}
    r0 = res[0]
    pb, rb, fb, sup = precision_recall_fscore_support(y, r0["oof_flat"], labels=classes,
                                                      zero_division=0)
    ph, rh, fh, _ = precision_recall_fscore_support(y, r0["oof_hier"], labels=classes,
                                                    zero_division=0)
    dd = pd.DataFrame({"환자수": sup, "F1_평면": fb.round(4), "F1_계층": fh.round(4),
                       "변화": (fh - fb).round(4),
                       "정밀도_평면": pb.round(3), "정밀도_계층": ph.round(3),
                       "재현율_평면": rb.round(3), "재현율_계층": rh.round(3)}, index=classes)
    dd["묶임"] = ["○" if c in touched else "" for c in classes]
    print(f"\n[7] 클래스별 변화 (cv_seed {r0['cv_seed']})")
    print("\n  묶인 클래스")
    print(dd[dd["묶임"] == "○"].to_string())
    other = dd[dd["묶임"] == ""].sort_values("변화")
    print("\n  안 묶인 클래스 중 가장 많이 내린 5개")
    print(other.head(5)[["환자수", "F1_평면", "F1_계층", "변화"]].to_string())
    print("\n  안 묶인 클래스 중 가장 많이 오른 5개")
    print(other.tail(5)[["환자수", "F1_평면", "F1_계층", "변화"]].to_string())
    print(f"\n  묶인 클래스 합계 변화   {dd[dd['묶임']=='○']['변화'].sum():+.4f}")
    print(f"  안 묶인 클래스 합계 변화 {other['변화'].sum():+.4f}")
    print("  → 편향 보정 때처럼 두 합계가 상쇄되면 이것도 제로섬이다.")

    # ── 저장 ──────────────────────────────────────────────────────────
    if not a.smoke:
        art = root / "experiments" / "iljun" / "exp_002_variant_type" / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        tab.to_csv(art / "hier_by_seed.csv", index=False, encoding="utf-8-sig")
        dd.to_csv(art / "hier_class_change.csv", encoding="utf-8-sig")
        pd.DataFrame(prow).to_csv(art / "hier_discovered_pairs.csv", index=False,
                                  encoding="utf-8-sig")
        if len(s2):
            s2.to_csv(art / "hier_stage2_folds.csv", index=False, encoding="utf-8-sig")
        (art / "hierarchical.json").write_text(json.dumps({
            "blocks": "".join(blocks), "model": model_key, "model_parameters": mp,
            "validation": pa.validation_spec(n_splits, model_seed, cv_seeds),
            "pair_discovery": {
                "method": "fold-train-only class centroid cosine similarity",
                "top_k": a.top_k,
                "rule_note": "라벨 이름을 코드에 쓰지 않는다. 쌍은 각 fold 의 학습 "
                             "분할에서만 산출하고 validation 에는 적용만 한다 (대회 규칙 2번).",
                "per_fold": allp,
                "frequency": {f"{x}+{z}": f"{c}/{n_folds}" for (x, z), c in cnt.most_common()},
            },
            "flat":   {"f1_macro": round(mf, 5), "f1_macro_std": round(sf, 5),
                       "per_seed": flat},
            "hierarchical": {"f1_macro": round(mh, 5), "f1_macro_std": round(sh, 5),
                             "per_seed": hier},
            "oracle": {"f1_macro": round(mo, 5), "f1_macro_std": round(so, 5),
                       "per_seed": orac,
                       "note": "2단계가 완벽하다고 가정한 상한. 실제는 이보다 낮다."},
            "stage1_f1": {r["cv_seed"]: r["stage1_f1"] for r in res},
            "stage2_standalone": (agg2.reset_index().to_dict("records") if len(s2) else []),
            "paired_delta": {"mean": round(m, 5), "std": round(sd, 5),
                             "positive_seeds": f"{pos}/{nn}"},
            "detected": bool(detected),
            "fingerprint": pa.fingerprint(),
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print("\n저장: artifacts/hierarchical.json · hier_by_seed.csv · "
              "hier_discovered_pairs.csv · hier_class_change.csv · hier_stage2_folds.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
