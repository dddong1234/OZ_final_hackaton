"""
exp_008-c · 저신호 클래스 one-vs-rest contrast
================================================

    .venv/bin/python3 experiments/iljun/exp_008_auto_confusion_pairs/run_class_contrast.py
    .venv/bin/python3 experiments/iljun/exp_008_auto_confusion_pairs/run_class_contrast.py --weak-k 5,8 --genes 5
    .venv/bin/python3 experiments/iljun/exp_008_auto_confusion_pairs/run_class_contrast.py --smoke

----------------------------------------------------------------------
왜 하나 — (b) 가 알려준 것
----------------------------------------------------------------------
혼동 지도를 만들어 보니 THYM 이 PCPG·PRAD·THCA·LAML·SARC **5개와 동시에** 헷갈렸다.
쌍(pair) 방식은 THYM 의 신호를 5조각으로 쪼갠다 — 모양이 안 맞는다.
이런 클래스엔 **one-vs-rest**(그 클래스 대 나머지 전부)가 맞는 형태다.

또 Macro F1 은 26개 클래스 F1 의 단순 평균이라 **못 맞히는 클래스가 점수를 끈다**.
잘하는 클래스는 이미 천장이라 더 올릴 게 없고, 약한 클래스 몇 개만 올려도 평균이
크게 움직인다. 그래서 '약한 클래스'를 fold-train 에서 찾아 거기에만 요약 피처를 준다.

----------------------------------------------------------------------
무엇을 만드나 (K 블록)
----------------------------------------------------------------------
각 fold 학습분할에서:
  1. 가벼운 대리모델(G 블록, 3-fold)로 클래스별 F1 을 재고 **가장 약한 K 개**를 고른다
  2. 그 클래스마다  contrast(gene) = rate(그 클래스) − rate(나머지 전부)
     |contrast| 상위 M 개 유전자를 고른다 (등장 10회 이상만 후보)
  3. 피처 2개 추가: 선택 유전자 변이수 합계 · 부호 반영 contrast 점수
피처는 클래스당 2개뿐이라 K=5 면 10개. 아주 얇다.

pairwise contrast(gs 원본)와 같은 계산 방식이고, '무엇과 대비하나'만 다르다.
  gs      : 왼쪽 암종 vs 오른쪽 암종
  여기(K) : 그 암종 vs 나머지 전부

----------------------------------------------------------------------
무엇을 비교하나
----------------------------------------------------------------------
  manual2            1등 구성 그대로 (기준선)
  manual2+classK     기준선 + K 블록          ← K 블록 단독 기여
  auto8              (b) 승자: 자동 8쌍
  auto8+classK       (b) 위에 K 블록          ← 두 축이 쌓이나
2단계(seed 42 탐색 → 최고 1개만 3-seed 확정)로 시간을 아낀다.

fold-train 라벨만 쓴다. test 는 열지 않는다. 클래스 이름·외부 지식은 안 본다.
gs 원본 파일은 수정하지 않는다(상속으로 K 블록만 덧붙인다).
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
from scipy import sparse
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
    hits = sorted(root.glob("experiments/gs/**/exp-gs-002-memory-safe.py"))
    if not hits:
        raise FileNotFoundError("exp-gs-002-memory-safe.py 없음 — main 을 pull 했나요?")
    spec = importlib.util.spec_from_file_location("gs_pipeline", hits[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules["gs_pipeline"] = module
    spec.loader.exec_module(module)
    return module


def surrogate(cache, idx, labels, seed, inner_splits=3, max_iter=300):
    """fold-train 안 3-fold 대리모델(G 블록) 1회로 두 가지를 동시에 얻는다.

    반환: (혼동 큰 쌍 내림차순, 클래스별 F1 오름차순)  — 둘 다 fold-train 라벨만 사용
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
    pred = np.array(list(pred))
    cm = confusion_matrix(y, pred, labels=classes)
    sizes = cm.sum(axis=1)
    pairs = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            swapped = cm[i, j] + cm[j, i]
            pairs.append((float(swapped / max(sizes[i] + sizes[j], 1)),
                          str(classes[i]), str(classes[j])))
    pairs.sort(key=lambda t: -t[0])
    per_class = f1_score(y, pred, average=None, labels=classes, zero_division=0)
    weak = sorted(zip(per_class, [str(c) for c in classes]), key=lambda t: t[0])
    return pairs, weak


def make_builder(gs):
    """gs.FoldMatrixBuilder 상속 — K 블록(클래스 one-vs-rest contrast)만 덧붙인다."""

    class ClassContrastBuilder(gs.FoldMatrixBuilder):
        def __init__(self, *args, weak_classes=(), class_genes: int = 5, **kwargs):
            super().__init__(*args, **kwargs)
            self.weak_classes = tuple(weak_classes)
            self.class_genes = class_genes

        def _domain_matrix(self, train_index, labels):
            base, names = super()._domain_matrix(train_index, labels)
            if not self.weak_classes:
                return base, names
            y = np.asarray(labels)[train_index]
            M = self.cache.mutation_matrix
            Mtr = M[train_index]
            support = np.asarray(Mtr.getnnz(axis=0)).ravel()
            cols, new_names = [], []
            for cls in self.weak_classes:
                mask = y == cls
                if not mask.any() or mask.all():
                    continue
                rate_in = np.asarray(Mtr[mask].getnnz(axis=0)).ravel() / mask.sum()
                rate_out = np.asarray(Mtr[~mask].getnnz(axis=0)).ravel() / (~mask).sum()
                contrast = rate_in - rate_out
                eligible = np.flatnonzero(support >= 10)
                selected = sorted(
                    eligible,
                    key=lambda i: (-abs(contrast[i]), -support[i], self.cache.genes[i])
                )[:self.class_genes]
                if not selected:
                    continue
                signs = np.sign(contrast[selected]).astype(np.float32)
                cols.append(sparse.csr_matrix(M[:, selected].sum(axis=1)))
                cols.append(M[:, selected].dot(sparse.csr_matrix(signs).T))
                new_names += [f"K__{cls}_count", f"K__{cls}_contrast"]
            if not cols:
                return base, names
            return sparse.hstack([base] + cols, format="csr"), names + new_names

    return ClassContrastBuilder


def eval_seed(gs, Builder, cache, labels, candidate, seed, n_splits, variants,
              max_pairs, max_weak, class_genes, verbose=True):
    """한 seed 에서 여러 variant 를 같은 fold 분할로. 대리모델은 fold 당 1회만."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = {v: np.empty(len(labels), dtype=object) for v in variants}
    nfeat, nwarn = {}, {v: 0 for v in variants}
    diag = {}
    for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        pairs_found, weak_found = surrogate(cache, tr, labels, seed)
        if fold == 0:
            diag = {"pairs": pairs_found[:max_pairs], "weak": weak_found[:max_weak]}
        for v in variants:
            use_auto = "auto" in v
            n_weak = int(v.split("class")[1]) if "class" in v else 0
            pairs = (tuple((l, r, 5) for _, l, r in pairs_found[:max_pairs])
                     if use_auto else candidate.contrast_pairs)
            weak = tuple(c for _, c in weak_found[:n_weak])
            builder = Builder(
                cache, candidate.backbone, candidate.exact_events, candidate.gene_pairs,
                candidate.gene_groups, candidate.hotspot_top_k, pairs,
                candidate.amino_mode, candidate.log1p_counts, candidate.b_count_binning,
                weak_classes=weak, class_genes=class_genes)
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
            print(f"    {v:20} {scores[v]:.5f}  (피처 {nfeat[v]}, 경고 {nwarn[v]})", flush=True)
    return scores, nfeat, diag


def paired(after: dict, before: dict):
    seeds = sorted(set(after) & set(before))
    d = np.array([after[s] - before[s] for s in seeds], dtype=float)
    sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    return round(float(d.mean()), 5), round(sd, 5), int((d > 0).sum()), len(d)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="저신호 클래스 one-vs-rest contrast")
    ap.add_argument("--weak-k", default="5,8", help="약한 클래스 개수 목록")
    ap.add_argument("--genes", type=int, default=5, help="클래스당 contrast 유전자 수")
    ap.add_argument("--auto-pairs", type=int, default=8,
                    help="(b) 승자 재현용 자동 쌍 개수. 0 이면 auto 변형 제외")
    ap.add_argument("--seeds", default="42,52,62")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    root = find_root(_HERE)
    gs = load_gs(root)
    Builder = make_builder(gs)
    seeds = [int(s) for s in a.seeds.split(",")][: (2 if a.smoke else None)]
    weak_ks = [int(k) for k in a.weak_k.split(",")][: (1 if a.smoke else None)]
    n_splits = 2 if a.smoke else gs.CONFIG.n_splits

    train = pd.read_csv(root / "data" / "raw" / "train.csv")
    gene_cols = [c for c in train.columns
                 if c not in (gs.CONFIG.id_col, gs.CONFIG.target_col)]
    if a.smoke:
        train = train.groupby(gs.CONFIG.target_col, group_keys=False).head(12).reset_index(drop=True)
        print("⚠ SMOKE — 클래스당 12행. 점수 의미 없음.")
    labels = train[gs.CONFIG.target_col]
    candidate = gs.CANDIDATES["H-AS-LR-exact-confusion-pairs-Apair-log1p"]

    variants = ["manual2"] + [f"manual2+class{k}" for k in weak_ks]
    if a.auto_pairs:
        variants += [f"auto{a.auto_pairs}"] + [f"auto{a.auto_pairs}+class{k}" for k in weak_ks]

    print("=" * 92)
    print("  exp_008-c · 저신호 클래스 one-vs-rest contrast (K 블록)")
    print(f"  기준 후보 {candidate.experiment_id}")
    print(f"  약한 클래스 {weak_ks} · 클래스당 유전자 {a.genes} · 자동쌍 {a.auto_pairs}")
    print("=" * 92)

    print("\nRowCache 생성...", flush=True)
    t0 = time.time()
    cache = gs.RowCache.build(train, gene_cols, show_progress=False)
    print(f"  완료 ({time.time() - t0:.0f}s)")

    max_weak = max(weak_ks)
    per_seed = {v: {} for v in variants}

    print(f"\n[1단계] 탐색 — seed {seeds[0]} 로 {len(variants)}개 변형")
    t0 = time.time()
    s0 = seeds[0]
    scores, nfeat, diag = eval_seed(gs, Builder, cache, labels, candidate, s0, n_splits,
                                    variants, a.auto_pairs, max_weak, a.genes)
    for v in variants:
        per_seed[v][s0] = scores[v]
    print(f"  ({time.time() - t0:.0f}s)")

    print("\n  fold 0 · 대리모델이 뽑은 '가장 약한 클래스' (F1 낮은 순)")
    for rank, (score, cls) in enumerate(diag["weak"], 1):
        print(f"     {rank:2}. {cls:8} F1 {score:.3f}")
    print("\n  fold 0 · 혼동 큰 쌍")
    for rank, (score, l, r) in enumerate(diag["pairs"][:5], 1):
        print(f"     {rank:2}. {l}↔{r}  {score:.3f}")

    others = [v for v in variants if v != "manual2"]
    best = max(others, key=lambda v: scores[v])
    print(f"\n  1단계 최고: {best} ({scores[best]:.5f}) vs manual2 ({scores['manual2']:.5f}) "
          f"→ {scores[best] - scores['manual2']:+.5f}")

    final = ["manual2", best]
    for s in seeds[1:]:
        print(f"\n[2단계] seed {s}")
        t0 = time.time()
        sc, _, _ = eval_seed(gs, Builder, cache, labels, candidate, s, n_splits,
                             final, a.auto_pairs, max_weak, a.genes)
        for v in final:
            per_seed[v][s] = sc[v]
        print(f"  ({time.time() - t0:.0f}s)")

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
    print("  → 전 seed 양수 + σ 초과. 채택 근거 있음." if ok else
          "  → σ 안이거나 seed 가 갈린다. 채택 근거 부족.")
    print("  ※ seed42 는 변형 선택에 쓴 seed 라 낙관 편향이 있다. "
          "나머지 seed 의 값을 따로 보라.")
    print(f"  예상 LB 이득 ~{dm * 0.64:+.4f} · 1위와 격차 0.00223")

    if not a.smoke:
        art = _HERE / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(art / "class_contrast.csv", index=False, encoding="utf-8-sig")
        (art / "class_contrast.json").write_text(json.dumps({
            "base_candidate": candidate.experiment_id,
            "weak_k": weak_ks, "class_genes": a.genes, "auto_pairs": a.auto_pairs,
            "seeds": seeds, "n_splits": n_splits,
            "stage1": scores, "best_variant": best, "per_seed": per_seed,
            "final": {v: {"f1_macro": ms(v)[0], "std": ms(v)[1]} for v in final},
            "best_vs_manual2": {"mean": dm, "std": dsd, "pos": f"{pos}/{nn}",
                                "detected": bool(ok)},
            "weak_classes_fold0": [[round(s, 4), c] for s, c in diag["weak"]],
            "pairs_fold0": [[round(s, 4), l, r] for s, l, r in diag["pairs"]],
            "table": rows,
            "note": "K 블록 = 클래스 vs 나머지 contrast. fold-train 도출, 규칙 안전. "
                    "gs 원본 무수정(상속으로 덧붙임).",
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print("\n저장: artifacts/class_contrast.json · class_contrast.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
