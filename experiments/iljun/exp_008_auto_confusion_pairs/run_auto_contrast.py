"""
exp_008 · 혼동 암종쌍 contrast 를 데이터로 자동 확장
=====================================================

    # ① 먼저 이것만 (몇 분) — 어떤 쌍이 자동으로 발견되는지 눈으로 확인
    .venv/bin/python3 experiments/iljun/exp_008_auto_confusion_pairs/run_auto_contrast.py --discover-only

    # ② 본 실험 (오래 걸림, 백그라운드 권장)
    .venv/bin/python3 experiments/iljun/exp_008_auto_confusion_pairs/run_auto_contrast.py
    .venv/bin/python3 experiments/iljun/exp_008_auto_confusion_pairs/run_auto_contrast.py --method cosine
    .venv/bin/python3 experiments/iljun/exp_008_auto_confusion_pairs/run_auto_contrast.py --smoke

----------------------------------------------------------------------
왜 하나
----------------------------------------------------------------------
1등 구성(exp-gs-002-08, LB 0.38711)의 contrast 블록은 **손으로 지정한 2쌍**
(KIRC↔KIPAN, LGG↔GBMLGG)뿐인데 그것만으로 0.433479 → 0.438495 (+0.005)였다.
26개 클래스에서 혼동쌍이 2개일 리 없다 — 나머지는 미개척이다.

여기서는 쌍을 **각 fold 학습 분할에서 자동 발견**해 N개로 확장한다.
덤: 코호트 이름(KIPAN⊃KIRC)을 사람이 지목하는 대신 데이터가 고르므로 대회
규칙 3(코호트 명칭 관계를 모델 구조에 반영 금지)에서도 안전해진다.

----------------------------------------------------------------------
쌍을 어떻게 찾나 — 두 방법
----------------------------------------------------------------------
  cosine     클래스별 변이 프로파일 중심의 코사인 유사도 상위 N.
             싸다. 단 '닮음'이 곧 '혼동'은 아닐 수 있다.
  confusion  fold-train 안에서 다시 3-fold 를 돌려(유전자 이진 G 블록만 쓰는
             가벼운 대리모델) 실제 오분류 혼동행렬을 만들고, 서로 많이 헷갈린
             쌍 상위 N. 느리지만 '혼동'의 문자 그대로의 정의다.  ← 기본

두 방법 모두 fold-train 라벨만 쓴다. test 는 열지 않는다. 코호트 이름은 안 본다.

★ 검증 포인트: 자동 발견이 상위에서 KIRC↔KIPAN / LGG↔GBMLGG 를 **다시 찾아내면**
  방법이 옳다는 증거다. 그 위에 더 붙는 쌍이 새 영역이다.
  --discover-only 로 모델 학습 없이 이것만 먼저 확인할 수 있다.

----------------------------------------------------------------------
2단계 구조 (시간 절약)
----------------------------------------------------------------------
  1단계 탐색   seed 42 하나로 manual2(기준) vs auto{N} 들을 비교 → 최고 1개 선별
  2단계 확정   선별된 것 + 기준선만 나머지 seed 로 채워 3-seed paired 판정
exp_006 에서 쓴 패턴. 전 조합을 3-seed 로 돌리면 몇 시간이라 이렇게 나눈다.

★ CV↔LB: 이 라인 gap 은 팀 실측 ~0.092. 1위와 격차 0.00223 이므로 CV 로 약
  +0.0035 가 필요하다. σ 안의 이득은 신뢰하지 않는다(gs-002-13 이 CV +0.004
  올리고 LB 는 -0.0004 였던 전례).

gs 원본 파일은 수정하지 않는다. contrast 피처 생성 코드는 gs 것을 그대로 쓰고
**쌍 선택만** 바꾸므로, 차이는 오직 '어떤 쌍을 고르나' 하나다.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

_HERE = Path(__file__).resolve().parent


def find_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "data" / "raw" / "train.csv").exists():
            return path
    raise FileNotFoundError("data/raw/train.csv 를 찾지 못했습니다")


def load_gs(root: Path):
    """gs 우승 파이프라인 로드(파일명에 '-' 가 있어 일반 import 불가)."""
    hits = sorted(root.glob("experiments/gs/**/exp-gs-002-memory-safe.py"))
    if not hits:
        raise FileNotFoundError("exp-gs-002-memory-safe.py 없음 — main 을 pull 했나요?")
    spec = importlib.util.spec_from_file_location("gs_pipeline", hits[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules["gs_pipeline"] = module
    spec.loader.exec_module(module)
    return module


# ── 쌍 발견 ────────────────────────────────────────────────────────────
def discover_cosine(cache, idx, labels, top_n):
    """클래스 중심 코사인 유사도 상위 N 쌍. 전체 평균을 빼 '흔한 성분'을 지운다."""
    y = np.asarray(labels)[idx]
    M = cache.mutation_matrix[idx]
    classes = np.unique(y)
    cent = np.vstack([np.asarray(M[y == c].mean(axis=0)).ravel() for c in classes])
    cent = cent - cent.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(cent, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    sim = (cent / norm) @ (cent / norm).T
    out = [(float(sim[i, j]), str(classes[i]), str(classes[j]))
           for i in range(len(classes)) for j in range(i + 1, len(classes))]
    out.sort(key=lambda t: -t[0])
    return out[:top_n]


def discover_confusion(cache, idx, labels, top_n, seed, inner_splits=3, max_iter=300):
    """fold-train 안에서 3-fold 대리모델(G 블록)을 돌려 실제 혼동이 큰 쌍 상위 N.

    대리모델은 유전자 이진화만 쓴다 — 쌍 후보를 고르는 용도라 가벼우면 충분하고,
    본 모델과 같은 피처를 다시 만들 필요가 없다. fold-train 라벨만 사용한다.
    """
    y = np.asarray(labels)[idx]
    X = cache.mutation_matrix[idx]
    classes = np.unique(y)
    pred = np.empty(len(y), dtype=object)
    inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    for itr, iva in inner.split(np.zeros(len(y)), y):
        model = LogisticRegression(solver="lbfgs", C=0.07, max_iter=max_iter,
                                   class_weight="balanced", random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X[itr], y[itr])
        pred[iva] = model.predict(X[iva])
    cm = confusion_matrix(y, np.array(list(pred)), labels=classes)
    sizes = cm.sum(axis=1)
    out = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            swapped = cm[i, j] + cm[j, i]                     # 서로 헷갈린 횟수
            denom = max(sizes[i] + sizes[j], 1)
            out.append((float(swapped / denom), str(classes[i]), str(classes[j])))
    out.sort(key=lambda t: -t[0])
    return out[:top_n]


DISCOVERY = {"cosine": discover_cosine, "confusion": discover_confusion}


def discover(method, cache, idx, labels, top_n, seed):
    if method == "cosine":
        return discover_cosine(cache, idx, labels, top_n)
    return discover_confusion(cache, idx, labels, top_n, seed)


MANUAL = {("KIRC", "KIPAN"), ("KIPAN", "KIRC"), ("LGG", "GBMLGG"), ("GBMLGG", "LGG")}


def show_pairs(found, prefix="   "):
    for rank, (score, left, right) in enumerate(found, 1):
        hit = " ★손지정" if (left, right) in MANUAL else ""
        print(f"{prefix}{rank:2}. {left}↔{right}  {score:.4f}{hit}")


# ── 평가 ───────────────────────────────────────────────────────────────
def eval_seed(gs, cache, labels, candidate, seed, n_splits, variants,
              methods, max_n, verbose=True):
    """한 seed 에서 여러 variant 를 같은 fold 분할로 평가. 발견은 fold 당 1회 재사용."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = {v: np.empty(len(labels), dtype=object) for v in variants}
    nfeat, nwarn, first_found = {}, {v: 0 for v in variants}, {}
    for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        found = {m: discover(m, cache, tr, labels, max_n, seed) for m in methods}
        if fold == 0:
            first_found = found
        for v in variants:
            if v == "manual2":
                pairs = candidate.contrast_pairs        # gs 원본: 쌍당 유전자 5 고정
            else:
                method, npart, gpart = v.split("_")     # 예: confusion_p8_g10
                n, g = int(npart[1:]), int(gpart[1:])
                pairs = tuple((l, r, g) for _, l, r in found[method][:n])
            builder = gs.FoldMatrixBuilder(
                cache, candidate.backbone, candidate.exact_events, candidate.gene_pairs,
                candidate.gene_groups, candidate.hotspot_top_k, pairs,
                candidate.amino_mode, candidate.log1p_counts, candidate.b_count_binning)
            Xtr, Xva, names = builder.build(tr, va, labels)
            model = gs.make_model("logistic", seed, candidate.lr_max_iter)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(Xtr, labels.iloc[tr])
            pred[v][va] = model.predict(Xva)
            nwarn[v] += sum(issubclass(c.category, ConvergenceWarning) for c in caught)
            nfeat[v] = len(names)
    scores = {v: round(float(f1_score(labels, np.array(list(pred[v])),
                                      average="macro", zero_division=0)), 5)
              for v in variants}
    if verbose:
        for v in variants:
            print(f"    {v:14} {scores[v]:.5f}  (피처 {nfeat[v]}, 경고 {nwarn[v]})", flush=True)
    return scores, nfeat, nwarn, first_found


def paired(after: dict, before: dict):
    seeds = sorted(set(after) & set(before))
    d = np.array([after[s] - before[s] for s in seeds], dtype=float)
    sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    return round(float(d.mean()), 5), round(sd, 5), int((d > 0).sum()), len(d)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="혼동쌍 contrast 자동 확장")
    ap.add_argument("--method", default="confusion", choices=("confusion", "cosine", "both"))
    ap.add_argument("--auto-n", default="2,5,8")
    ap.add_argument("--genes", default="5", help="쌍당 contrast 유전자 수 (콤마 목록 가능). gs 기본 5")
    ap.add_argument("--seeds", default="42,52,62")
    ap.add_argument("--discover-only", action="store_true",
                    help="모델 비교 없이 어떤 쌍이 발견되는지만 출력(빠름)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    root = find_root(_HERE)
    gs = load_gs(root)
    methods = ["confusion", "cosine"] if a.method == "both" else [a.method]
    seeds = [int(s) for s in a.seeds.split(",")][: (2 if a.smoke else None)]
    auto_ns = [int(n) for n in a.auto_n.split(",")][: (1 if a.smoke else None)]
    gene_list = [int(g) for g in str(a.genes).split(",")][: (1 if a.smoke else None)]
    n_splits = 2 if a.smoke else gs.CONFIG.n_splits

    train = pd.read_csv(root / "data" / "raw" / "train.csv")
    genes_cols = [c for c in train.columns
                  if c not in (gs.CONFIG.id_col, gs.CONFIG.target_col)]
    if a.smoke:
        train = train.groupby(gs.CONFIG.target_col, group_keys=False).head(12).reset_index(drop=True)
        print("⚠ SMOKE — 클래스당 12행. 점수 의미 없음.")
    labels = train[gs.CONFIG.target_col]
    candidate = gs.CANDIDATES["H-AS-LR-exact-confusion-pairs-Apair-log1p"]

    print("=" * 92)
    print("  exp_008 · 혼동쌍 contrast 자동 확장 (1등 구성 위에서)")
    print(f"  기준 후보 {candidate.experiment_id}")
    print(f"  발견 방법 {methods} · 쌍 개수 {auto_ns} · 쌍당 유전자 {gene_list}")
    print("=" * 92)

    print("\nRowCache 생성 (한 번만)...", flush=True)
    t0 = time.time()
    cache = gs.RowCache.build(train, genes_cols, show_progress=False)
    print(f"  완료 ({time.time() - t0:.0f}s)")

    # ── 발견만 보고 끝내기 ────────────────────────────────────────────
    if a.discover_only:
        print("\n[발견 진단] 전체 train 기준 상위 10쌍 — 손지정 쌍을 다시 찾는지 확인")
        print("  (진단 전용. 본 실험에서는 fold-train 에서만 발견한다)")
        idx = np.arange(len(labels))
        out = {}
        for m in methods:
            t0 = time.time()
            found = discover(m, cache, idx, labels, 10, seeds[0])
            out[m] = found
            print(f"\n  ── {m} ({time.time() - t0:.0f}s)")
            show_pairs(found)
            hits = sum(1 for _, l, r in found if (l, r) in MANUAL)
            print(f"     → 손지정 2쌍 중 {hits}개를 상위 10 안에서 재발견")
        art = _HERE / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        (art / "discovery.json").write_text(json.dumps(
            {m: [[round(s, 4), l, r] for s, l, r in v] for m, v in out.items()},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n저장: artifacts/discovery.json")
        print("손지정 쌍이 상위에 나오면 그 방법으로 본 실험을 돌리세요.")
        return 0

    variants = ["manual2"] + [f"{m}_p{n}_g{g}"
                              for m in methods for n in auto_ns for g in gene_list]
    max_n = max(auto_ns)
    per_seed = {v: {} for v in variants}

    # ── 1단계: seed 42 탐색 ───────────────────────────────────────────
    print(f"\n[1단계] 탐색 — seed {seeds[0]} 로 {len(variants)}개 변형 비교")
    t0 = time.time()
    s0 = seeds[0]
    scores, nfeat, nwarn, found0 = eval_seed(
        gs, cache, labels, candidate, s0, n_splits, variants, methods, max_n)
    for v in variants:
        per_seed[v][s0] = scores[v]
    print(f"  ({time.time() - t0:.0f}s)")

    print("\n  fold 0 에서 발견된 쌍 (상위 %d)" % max_n)
    for m, found in found0.items():
        print(f"  ── {m}")
        show_pairs(found[:max_n], prefix="     ")

    auto_only = [v for v in variants if v != "manual2"]
    best = max(auto_only, key=lambda v: scores[v])
    print(f"\n  1단계 최고 자동 변형: {best} ({scores[best]:.5f}) "
          f"vs manual2 ({scores['manual2']:.5f}) → {scores[best]-scores['manual2']:+.5f}")

    # ── 2단계: 최고 1개 + 기준선만 3-seed 확정 ────────────────────────
    final = ["manual2", best]
    rest = seeds[1:]
    if rest:
        print(f"\n[2단계] 확정 — {final} 를 seed {rest} 로 채워 3-seed paired 판정")
        for s in rest:
            t0 = time.time()
            sc, _, _, _ = eval_seed(gs, cache, labels, candidate, s, n_splits,
                                    final, methods, max_n)
            for v in final:
                per_seed[v][s] = sc[v]
            print(f"  seed {s} 완료 ({time.time() - t0:.0f}s)")

    def ms(v):
        arr = np.array(list(per_seed[v].values()), dtype=float)
        return round(arr.mean(), 5), (round(arr.std(ddof=1), 5) if len(arr) > 1 else None)

    print("\n" + "=" * 92)
    print("  최종 (3-seed paired)")
    print("=" * 92)
    rows = []
    for v in final:
        m, sd = ms(v)
        if v == "manual2":
            rows.append({"변형": v, "Macro F1": m, "σ": sd, "기준 대비": "—", "양수seed": "—"})
        else:
            dm, dsd, pos, nn = paired(per_seed[v], per_seed["manual2"])
            rows.append({"변형": v, "Macro F1": m, "σ": sd,
                         "기준 대비": f"{dm:+.5f}", "양수seed": f"{pos}/{nn}"})
    print(pd.DataFrame(rows).to_string(index=False))

    dm, dsd, pos, nn = paired(per_seed[best], per_seed["manual2"])
    ok = (pos == nn and nn > 1) and abs(dm) >= (dsd if np.isfinite(dsd) and dsd > 0 else 0)
    print(f"\n판정: {best} − manual2 = {dm:+.5f} ± {dsd} ({pos}/{nn} seed 양수)")
    if ok:
        print("  → 전 seed 양수 + σ 초과. 채택 근거 있음.")
    elif pos == nn and nn > 1:
        print("  → 방향은 일관되나 σ 안. 제출로만 확인 가능.")
    else:
        print("  → seed 가 갈린다. 자동 확장 이득 없음(손 지정 2쌍으로 충분).")
    print(f"  예상 LB 이득 ~{dm * 0.64:+.4f} (패스스루 64%) · 1위와 격차 0.00223")
    print("  ※ gs-002-13 이 CV +0.004 에도 LB -0.0004 였다. CV 최댓값 과신 금물.")

    if not a.smoke:
        art = _HERE / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(art / "auto_contrast.csv", index=False, encoding="utf-8-sig")
        (art / "auto_contrast.json").write_text(json.dumps({
            "base_candidate": candidate.experiment_id,
            "lr": {"C": gs.CONFIG.lr_c, "max_iter": gs.CONFIG.lr_max_iter},
            "n_splits": n_splits, "seeds": seeds, "methods": methods,
            "auto_n": auto_ns, "genes_per_pair": gene_list,
            "stage1_seed42": scores,
            "best_variant": best,
            "per_seed": per_seed,
            "final": {v: {"f1_macro": ms(v)[0], "std": ms(v)[1]} for v in final},
            "best_vs_manual2": {"mean": dm, "std": dsd, "pos": f"{pos}/{nn}", "detected": bool(ok)},
            "discovered_fold0": {m: [[round(s, 4), l, r] for s, l, r in f]
                                 for m, f in found0.items()},
            "table": rows,
            "note": "쌍 선택만 다르고 contrast 피처 코드는 gs 원본 재사용. "
                    "fold-train 도출이라 규칙 3 안전.",
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print("\n저장: artifacts/auto_contrast.json · auto_contrast.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
