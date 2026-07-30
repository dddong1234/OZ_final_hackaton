"""
트랙 A 검증 파이프라인 — exp_002_variant_type

팀 방식(training/run.py, common.experiment)은 holdout 으로 공식 결과(results/metrics.json)를
만든다. 이 파일은 그것과 별개로 **CV · 제출 게이트 · 지문 · 노트북 교차검증**을 담당한다.
결과는 results/metrics_cv.json 에 쓴다 (팀 metrics.json 을 덮지 않는다).

    python experiments/member_d/exp_002_variant_type/pipeline.py            # 정본(config.yaml)
    python experiments/member_d/exp_002_variant_type/pipeline.py --smoke    # 30초 배선 점검
    python experiments/member_d/exp_002_variant_type/pipeline.py --repeat 2 # 결정성 확인
    python experiments/member_d/exp_002_variant_type/pipeline.py --check-only

노트북에서도 같은 함수를 부른다 (두 경로가 갈라지지 않도록):
    from pipeline import cross_validate, run_pipeline

설계 철학 · 배제한 것은 features_A.py 상단 docstring 을 본다.
점수는 5자리 기록, 셋째 자리로 판정 (협업 규정 2). validation·model_parameters 형식은
GIT_STRATEGY §9.2 를 따른다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import features_A as fa                                          # noqa: E402
from features_A import build_features, fit_spec, parse_sample_counts, spec_to_json  # noqa: E402

PIPELINE_VERSION = "exp002_v1"
TARGET, ID = "SUBCLASS", "ID"

MODELS = {
    "logreg": ("LogisticRegression(balanced)",
               lambda seed, mp: LogisticRegression(random_state=seed, **mp)),
    "svm":    ("LinearSVC(balanced)",
               lambda seed, mp: LinearSVC(**mp)),
    "sgd":    ("SGD(modified_huber, balanced)",
               lambda seed, mp: SGDClassifier(loss="modified_huber", random_state=seed,
                                              n_jobs=-1, **mp)),
}
DEFAULT_MODEL_PARAMS = {
    "logreg": {"max_iter": 1000, "class_weight": "balanced"},
    "svm":    {"class_weight": "balanced", "max_iter": 3000},
    "sgd":    {"class_weight": "balanced", "max_iter": 3000},
}


def find_project_root(start: Path | None = None) -> Path:
    for path in [Path(start or Path.cwd()).resolve(), *_HERE.parents]:
        if (path / "configs" / "baseline.yaml").exists():
            return path
    raise FileNotFoundError("저장소 루트를 찾지 못했습니다. --root 로 지정하세요.")


def load_cfg() -> dict:
    return yaml.safe_load((_HERE / "config.yaml").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint() -> dict:
    return {
        "pipeline": f"pipeline.py@{PIPELINE_VERSION}", "pipeline_sha256": sha256(__file__),
        "features": f"features_A.py@{fa.__version__}", "features_sha256": sha256(fa.__file__),
        "preprocess_sha256": sha256(_HERE / "preprocessing" / "preprocess.py"),
    }


def validation_spec(n_splits: int, seed: int) -> dict:
    return {"method": "StratifiedKFold", "n_splits": int(n_splits),
            "shuffle": True, "seeds": [int(seed)]}


def verdict(f1: float, ref: float, dec: int = 3) -> str:
    a, b = round(float(f1), dec), round(float(ref), dec)
    return "[+] 향상" if a > b else ("[-] 하락" if a < b else "[=] 동일")


def _log(m, v=True):
    if v:
        print(m, flush=True)


# ----------------------------------------------------------------------
def load_data(root, smoke=False, v=True):
    d = Path(root) / "data" / "raw"
    train = pd.read_csv(d / "train.csv"); test = pd.read_csv(d / "test.csv")
    sub = pd.read_csv(d / "sample_submission.csv")
    if smoke:
        train = train.groupby(TARGET, group_keys=False).head(12).reset_index(drop=True)
        test = test.head(300).reset_index(drop=True); sub = sub.head(300).reset_index(drop=True)
        _log("[Step 1] ⚠ SMOKE — 클래스당 12행. 점수 의미 없음.", v)
    gene_cols = [c for c in train.columns if c not in (TARGET, ID)]
    _log(f"[Step 1] train {train.shape} · test {test.shape} · 유전자 {len(gene_cols)}", v)
    return train, test, sub, gene_cols


def parse_all(train, test, gene_cols, v=True):
    t = time.time()
    ct, cte = parse_sample_counts(train, gene_cols), parse_sample_counts(test, gene_cols)
    _log(f"[Step 2] 파싱 {time.time() - t:.0f}s", v)
    return ct, cte


def leakage_checks(train, test, cte, gene_cols, blocks, seed, v=True):
    spec = fit_spec(train, gene_cols, seed=seed)
    full, _ = build_features(test, cte, spec, blocks)
    head = test.iloc[:100]
    part, _ = build_features(head, parse_sample_counts(head, gene_cols), spec, blocks)
    one = test.iloc[[7]]
    onex, _ = build_features(one, parse_sample_counts(one, gene_cols), spec, blocks)
    checks = [
        ("부분집합 불변성 (앞 100행)", np.array_equal(part.toarray(), full[:100].toarray())),
        ("단일 행 독립성", np.array_equal(onex.toarray(), full[[7]].toarray())),
        ("spec 재현성", fit_spec(train, gene_cols, seed=seed)["keep_idx"] == spec["keep_idx"]),
        ("NaN·inf 없음", bool(np.isfinite(full.data).all())),
        ("test 결측 fillna 처리", int(test.isna().sum().sum()) > 0),
    ]
    for n_, ok in checks:
        _log(f"         [{'PASS' if ok else 'FAIL'}] {n_}", v)
    return all(o for _, o in checks), [{"check": n, "passed": bool(o)} for n, o in checks]


def cross_validate(train, y, counts, gene_cols, blocks, model_key="logreg",
                   model_params=None, seed=42, n_splits=5, label=None, v=True):
    """fold 안에서 spec 을 다시 fit 하는 정직한 CV. 노트북도 이 함수를 쓴다."""
    name, fn = MODELS[model_key]
    mp = model_params or DEFAULT_MODEL_PARAMS[model_key]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.empty(len(y), dtype=object); dim = 0; t = time.time()
    for i_tr, i_va in cv.split(train, y):
        spec = fit_spec(train.iloc[i_tr], gene_cols, seed=seed)
        Xa, _ = build_features(train.iloc[i_tr], counts.iloc[i_tr], spec, blocks)
        Xb, _ = build_features(train.iloc[i_va], counts.iloc[i_va], spec, blocks)
        oof[i_va] = fn(seed, mp).fit(Xa, y[i_tr]).predict(Xb); dim = Xa.shape[1]
    oof = np.array(list(oof))
    f1 = round(float(f1_score(y, oof, average="macro")), 5)
    acc = round(float(accuracy_score(y, oof)), 5)
    if v:
        print(f"         {(label or ''.join(blocks)):34} dim {dim:5d}  F1 {f1:.5f}  "
              f"Acc {acc:.5f}  ({time.time() - t:.0f}s)", flush=True)
    return {"blocks": "".join(blocks), "model": model_key, "model_name": name,
            "dim": int(dim), "f1_macro": f1, "accuracy": acc, "oof": oof}


def run_pipeline(root=None, blocks=None, model=None, repeat=1, smoke=False,
                 submit_gate=None, v=True, write=True) -> dict:
    cfg = load_cfg()
    p = cfg["pipeline"]
    root = Path(root) if root else find_project_root()
    blocks = blocks or p["blocks"]
    model = model or p["model"]
    seed = p["cv"]["seed"]
    n_splits = 2 if smoke else p["cv"]["n_splits"]
    dec = p["decimals"]
    gate = p["submit_gate"] if submit_gate is None else submit_gate
    ref = p["baseline"]
    mp = DEFAULT_MODEL_PARAMS[model]
    bt = tuple(blocks)
    name, fn = MODELS[model]
    exp_id = cfg["experiment"]["id"]

    _log("=" * 72, v)
    _log(f"  {exp_id} · pipeline {PIPELINE_VERSION} · features_A {fa.__version__}", v)
    _log(f"  피처 {blocks} · {name} · seed {seed} · StratifiedKFold-{n_splits}", v)
    _log(f"  기준선 {ref['experiment']} = {ref['f1_macro']:.5f} (판정 {round(ref['f1_macro'], dec)})", v)
    _log("=" * 72, v)

    train, test, submission, gene_cols = load_data(root, smoke, v)
    y = train[TARGET].values
    ct, cte = parse_all(train, test, gene_cols, v)

    ok, checks = leakage_checks(train, test, cte, gene_cols, bt, seed, v)
    if not ok:
        raise RuntimeError("Leakage 자가검증 실패")

    _log(f"[Step 4] 교차검증 {repeat}회", v)
    runs = [cross_validate(train, y, ct, gene_cols, bt, model, mp, seed, n_splits,
                           label=f"run {i + 1}/{repeat}", v=v) for i in range(repeat)]
    r0 = runs[0]
    det = len({r["f1_macro"] for r in runs}) == 1
    if repeat > 1:
        _log(f"         결정성 {'PASS' if det else 'FAIL'}", v)
    f1, acc = r0["f1_macro"], r0["accuracy"]
    gate_pass = round(f1, dec) > round(ref["f1_macro"], dec)

    _log("=" * 72, v)
    _log(f"  Macro F1 {f1:.5f} ({f1 - ref['f1_macro']:+.5f})  Acc {acc:.5f}  "
         f"{verdict(f1, ref['f1_macro'], dec)}", v)
    _log("=" * 72, v)

    res = {
        "experiment": exp_id, "owner": "member_d", "track": "A", "model": name,
        "seed": seed, "validation": validation_spec(n_splits, seed),
        "model_parameters": mp, "accuracy": acc, "f1_macro": f1,
        "feature_blocks": blocks, "n_features_cv": r0["dim"],
        "baseline": ref, "delta_vs_baseline": round(f1 - ref["f1_macro"], 5),
        "verdict": verdict(f1, ref["f1_macro"], dec),
        "cv_runs": [{"f1_macro": r["f1_macro"], "accuracy": r["accuracy"]} for r in runs],
        "deterministic": bool(det) if repeat > 1 else None,
        "leakage_checks": checks,
        "submission_gate": {"enabled": gate, "passed": bool(gate_pass),
                            "threshold": round(ref["f1_macro"], dec), "file": None},
        "fingerprint": fingerprint(),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "sklearn": sklearn.__version__},
        "note": "CV 검증 파이프라인. 팀 공식 결과(holdout)는 results/metrics.json (training.run).",
        "smoke": bool(smoke), "oof": r0["oof"],
    }

    if gate and not gate_pass:
        _log(f"[Step 5] GATE — 기준선 {round(ref['f1_macro'], dec)} 미달. submission 생략.", v)
    if smoke:
        return res

    spec_final = fit_spec(train, gene_cols, seed=seed)
    Xtr, feat_names = build_features(train, ct, spec_final, bt)
    res["n_features"] = int(Xtr.shape[1])
    sub = None
    if not (gate and not gate_pass):
        Xte, _ = build_features(test, cte, spec_final, bt)
        pred = fn(seed, mp).fit(Xtr, y).predict(Xte)
        subck = [("행 수", len(submission) == len(test)),
                 ("컬럼", list(submission.columns) == [ID, TARGET]),
                 ("결측 없음", int(pd.Series(pred).isna().sum()) == 0),
                 ("train 클래스 안", set(pred).issubset(set(y))),
                 ("쏠림 없음", pd.Series(pred).value_counts(normalize=True).iloc[0] < 0.40)]
        for n_, o_ in subck:
            _log(f"         [{'PASS' if o_ else 'FAIL'}] {n_}", v)
        res["submission_checks"] = [{"check": n, "passed": bool(o)} for n, o in subck]
        sub = submission.copy(); sub[TARGET] = pred

    if write:
        out = root / "experiments" / "member_d" / "exp_002_variant_type" / "results"
        out.mkdir(parents=True, exist_ok=True)
        if sub is not None:
            sp = out / f"submission_f1_{f1:.5f}.csv"
            sub.to_csv(sp, index=False, encoding="UTF-8-sig")
            res["submission_gate"]["file"] = sp.name
        (out / "metrics_cv.json").write_text(
            json.dumps({k: v_ for k, v_ in res.items() if k != "oof"},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        # metrics_cv.json 은 .gitignore 대상이라, CV 결과가 git 에 남도록
        # results/README.md(커밋 허용)에도 한눈 요약을 쓴다.
        fp = res["fingerprint"]
        (out / "README.md").write_text(
            f"# {exp_id} — 결과 요약\n\n"
            f"| 지표 | 값 |\n|---|---|\n"
            f"| CV Macro F1 (StratifiedKFold-{n_splits}, seed {seed}) | **{f1:.5f}** |\n"
            f"| CV Accuracy | {acc:.5f} |\n"
            f"| 기준선 {ref['experiment']} | {ref['f1_macro']:.5f} |\n"
            f"| 기준선 대비 | {res['delta_vs_baseline']:+.5f} · {res['verdict']} |\n"
            f"| 제출 게이트 | {'통과' if res['submission_gate']['passed'] else '미달'} "
            f"(기준 {res['submission_gate']['threshold']}) |\n\n"
            f"**지문** — pipeline `{fp['pipeline_sha256'][:12]}` · "
            f"features `{fp['features_sha256'][:12]}` · preprocess `{fp['preprocess_sha256'][:12]}`\n\n"
            f"> 상세(비커밋): `metrics_cv.json` · 팀 holdout: `metrics.json` (training.run)\n"
            f"> 재현: `python3 experiments/member_d/exp_002_variant_type/pipeline.py`\n",
            encoding="utf-8")
        _log(f"[Step 6] results/metrics_cv.json + results/README.md "
             f"(metrics_cv 는 gitignore, README·metrics.json 은 커밋)", v)

    _log("=" * 72, v)
    _log(f"  {res['verdict']}  {exp_id}  F1 {f1:.5f} ({res['delta_vs_baseline']:+.5f})  "
         f"Acc {acc:.5f}", v)
    _log(f"  지문 pipeline {res['fingerprint']['pipeline_sha256'][:12]} · "
         f"features {res['fingerprint']['features_sha256'][:12]}", v)
    _log("=" * 72, v)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="트랙 A 검증 파이프라인 (exp_002)")
    ap.add_argument("--root", default=None)
    ap.add_argument("--blocks", default=None)
    ap.add_argument("--model", default=None, choices=sorted(MODELS))
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root) if a.root else find_project_root()

    if a.check_only:
        cfg = load_cfg()["pipeline"]
        train, test, _, gene_cols = load_data(root, a.smoke)
        _, cte = parse_all(train, test, gene_cols)
        ok, _ = leakage_checks(train, test, cte, gene_cols,
                               tuple(a.blocks or cfg["blocks"]), cfg["cv"]["seed"])
        return 0 if ok else 1
    try:
        run_pipeline(root=root, blocks=a.blocks, model=a.model, repeat=a.repeat,
                     smoke=a.smoke, submit_gate=(False if a.no_gate else None))
    except RuntimeError as e:
        print(f"\n[FAIL] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
