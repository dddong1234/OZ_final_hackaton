"""
exp_008-d · 위치구간(position bin) 독립 기여 재검증
====================================================

    .venv/bin/python3 experiments/iljun/exp_008_auto_confusion_pairs/run_position_bins.py
    .venv/bin/python3 experiments/iljun/exp_008_auto_confusion_pairs/run_position_bins.py --smoke

----------------------------------------------------------------------
왜 하나 — 아무도 안 재본 6개
----------------------------------------------------------------------
gs 의 FE #37(exp-gs-002-07)이 A_all(426) → A_pair-only(380) 로 바꾸며 +0.040 을
얻었다. 그때 제거된 46개는 다음 셋을 **한꺼번에** 버린 것이다.

    ref 단독 20 + alt 단독 20 + **위치구간 6**

우리 exp_007 은 이 중 **marginal(ref/alt 40)이 해롭다(-0.03, 0/3 seed)** 는 걸
따로 확인했다. 그런데 **위치구간 6개만의 기여는 아무도 측정한 적이 없다** —
해로운 marginal 에 묻어서 같이 버려졌을 뿐이다.

피처 6개짜리라 비용도 리스크도 거의 0 이다. 남은 값싼 카드.

가설: 단백질의 어느 위치에 변이가 몰리는가는 치환 방향(pair)과 다른 정보다.
      (예: N말단 절단형 vs 중간 도메인 미스센스)  다만 근거는 약하다 — 아래 참고.
반대 근거: 우리 exp_005 에서 '코돈(위치) 축'은 정확변이 대비 이득이 없었다.
           위치 정보가 이 데이터에서 약할 수 있다.  판정은 데이터로.

----------------------------------------------------------------------
무엇을 비교하나
----------------------------------------------------------------------
  manual2        1등 구성 그대로 (A_pair-only + log1p)      ← 기준선
  manual2+pos    기준선 + 위치구간 6                        ← 위치 단독 기여
  auto8          (b) 승자: 자동 8쌍 contrast
  auto8+pos      (b) 위에 위치구간 6                        ← 두 축이 쌓이나
2단계(seed 42 탐색 → 최고 1개만 3-seed 확정).

위치구간은 amino[:, 420:426] (gs RowCache 레이아웃). count 형이라 log1p_counts
설정을 그대로 따른다 — gs 가 A_all 에 적용하던 처리와 동일하다.

fold-train 만 fit. test 미열람. gs 원본 무수정(상속으로 6열만 덧붙임).
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
POS_SLICE = slice(420, 426)          # ref20 + alt20 + pair380 = 420 이후가 위치구간 6


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


def discover_confusion(cache, idx, labels, top_n, seed, inner_splits=3, max_iter=300):
    """fold-train 3-fold 대리모델(G 블록)로 혼동 큰 쌍 상위 N. (b) 와 동일."""
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
            out.append((float((cm[i, j] + cm[j, i]) / max(sizes[i] + sizes[j], 1)),
                        str(classes[i]), str(classes[j])))
    out.sort(key=lambda t: -t[0])
    return out[:top_n]


def make_builder(gs):
    """gs.FoldMatrixBuilder 상속 — 위치구간 6열만 덧붙인다."""

    class PositionBuilder(gs.FoldMatrixBuilder):
        def __init__(self, *args, with_position: bool = False, **kwargs):
            super().__init__(*args, **kwargs)
            self.with_position = with_position

        def _domain_matrix(self, train_index, labels):
            base, names = super()._domain_matrix(train_index, labels)
            if not self.with_position:
                return base, names
            pos = self.cache.amino[:, POS_SLICE]
            if self.log1p_counts:                 # gs 가 A 블록에 하던 처리와 동일
                pos = np.log1p(pos)
            block = sparse.csr_matrix(pos.astype(np.float32))
            new_names = [f"P__position_bin_{i}" for i in range(block.shape[1])]
            return sparse.hstack([base, block], format="csr"), names + new_names

    return PositionBuilder


def eval_seed(gs, Builder, cache, labels, candidate, seed, n_splits, variants,
              auto_pairs, verbose=True):
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = {v: np.empty(len(labels), dtype=object) for v in variants}
    nfeat, nwarn, diag = {}, {v: 0 for v in variants}, []
    need_auto = any("auto" in v for v in variants)
    for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        found = discover_confusion(cache, tr, labels, auto_pairs, seed) if need_auto else []
        if fold == 0:
            diag = found
        for v in variants:
            pairs = (tuple((l, r, 5) for _, l, r in found)
                     if "auto" in v else candidate.contrast_pairs)
            builder = Builder(
                cache, candidate.backbone, candidate.exact_events, candidate.gene_pairs,
                candidate.gene_groups, candidate.hotspot_top_k, pairs,
                candidate.amino_mode, candidate.log1p_counts, candidate.b_count_binning,
                with_position=v.endswith("+pos"))
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
            print(f"    {v:16} {scores[v]:.5f}  (피처 {nfeat[v]}, 경고 {nwarn[v]})", flush=True)
    return scores, nfeat, diag


def paired(after: dict, before: dict):
    seeds = sorted(set(after) & set(before))
    d = np.array([after[s] - before[s] for s in seeds], dtype=float)
    sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    return round(float(d.mean()), 5), round(sd, 5), int((d > 0).sum()), len(d)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="위치구간 독립 기여 재검증")
    ap.add_argument("--auto-pairs", type=int, default=8, help="0 이면 auto 변형 제외")
    ap.add_argument("--seeds", default="42,52,62")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    root = find_root(_HERE)
    gs = load_gs(root)
    Builder = make_builder(gs)
    seeds = [int(s) for s in a.seeds.split(",")][: (2 if a.smoke else None)]
    n_splits = 2 if a.smoke else gs.CONFIG.n_splits

    train = pd.read_csv(root / "data" / "raw" / "train.csv")
    gene_cols = [c for c in train.columns
                 if c not in (gs.CONFIG.id_col, gs.CONFIG.target_col)]
    if a.smoke:
        train = train.groupby(gs.CONFIG.target_col, group_keys=False).head(12).reset_index(drop=True)
        print("⚠ SMOKE — 클래스당 12행. 점수 의미 없음.")
    labels = train[gs.CONFIG.target_col]
    candidate = gs.CANDIDATES["H-AS-LR-exact-confusion-pairs-Apair-log1p"]

    variants = ["manual2", "manual2+pos"]
    if a.auto_pairs:
        variants += [f"auto{a.auto_pairs}", f"auto{a.auto_pairs}+pos"]

    print("=" * 92)
    print("  exp_008-d · 위치구간(position bin) 6개의 독립 기여")
    print(f"  기준 후보 {candidate.experiment_id} (A_pair-only + log1p)")
    print(f"  위치구간 = amino[:, 420:426] · log1p {candidate.log1p_counts}")
    print("=" * 92)

    print("\nRowCache 생성...", flush=True)
    t0 = time.time()
    cache = gs.RowCache.build(train, gene_cols, show_progress=False)
    print(f"  완료 ({time.time() - t0:.0f}s) · amino {cache.amino.shape}")

    per_seed = {v: {} for v in variants}
    print(f"\n[1단계] 탐색 — seed {seeds[0]} 로 {len(variants)}개 변형")
    t0 = time.time()
    s0 = seeds[0]
    scores, nfeat, diag = eval_seed(gs, Builder, cache, labels, candidate, s0,
                                    n_splits, variants, a.auto_pairs)
    for v in variants:
        per_seed[v][s0] = scores[v]
    print(f"  ({time.time() - t0:.0f}s)")

    print("\n  위치 단독 기여 (manual2 대비): "
          f"{scores['manual2+pos'] - scores['manual2']:+.5f}")
    if a.auto_pairs:
        key = f"auto{a.auto_pairs}"
        print(f"  위치 추가 기여 ({key} 대비): "
              f"{scores[key + '+pos'] - scores[key]:+.5f}")

    others = [v for v in variants if v != "manual2"]
    best = max(others, key=lambda v: scores[v])
    print(f"\n  1단계 최고: {best} ({scores[best]:.5f}) vs manual2 ({scores['manual2']:.5f}) "
          f"→ {scores[best] - scores['manual2']:+.5f}")

    final = ["manual2", best]
    for s in seeds[1:]:
        print(f"\n[2단계] seed {s}")
        t0 = time.time()
        sc, _, _ = eval_seed(gs, Builder, cache, labels, candidate, s, n_splits,
                             final, a.auto_pairs)
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
    print("  ※ seed42 는 변형 선택에 쓴 seed. 나머지 seed 값을 따로 보라.")
    print(f"  예상 LB 이득 ~{dm * 0.64:+.4f} · 1위와 격차 0.00223")

    if not a.smoke:
        art = _HERE / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(art / "position_bins.csv", index=False, encoding="utf-8-sig")
        (art / "position_bins.json").write_text(json.dumps({
            "base_candidate": candidate.experiment_id,
            "position_slice": [POS_SLICE.start, POS_SLICE.stop],
            "auto_pairs": a.auto_pairs, "seeds": seeds, "n_splits": n_splits,
            "stage1": scores, "best_variant": best, "per_seed": per_seed,
            "position_alone_delta_seed42": round(
                scores["manual2+pos"] - scores["manual2"], 5),
            "final": {v: {"f1_macro": ms(v)[0], "std": ms(v)[1]} for v in final},
            "best_vs_manual2": {"mean": dm, "std": dsd, "pos": f"{pos}/{nn}",
                                "detected": bool(ok)},
            "pairs_fold0": [[round(s, 4), l, r] for s, l, r in diag],
            "table": rows,
            "note": "gs FE#37 이 marginal 40 과 함께 버린 위치구간 6 의 독립 기여를 분리 측정. "
                    "gs 원본 무수정(상속으로 6열 추가).",
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print("\n저장: artifacts/position_bins.json · position_bins.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
