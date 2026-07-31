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

PIPELINE_VERSION = "exp002_v3"
TARGET, ID = "SUBCLASS", "ID"

# 팀 common/preprocessing_benchmark.py 와 동일한 다중 seed 프로토콜.
# CV 폴드 분할 seed 만 바꾸고 모델 random_state 는 고정한다 — 그래야 팀
# 벤치마크(3-seed Logistic 0.33738 ± 0.00625)와 같은 잣대로 비교된다.
CONFIRMATION_CV_SEEDS = (42, 52, 62)
MODEL_SEED = 42

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


def validation_spec(n_splits: int, model_seed: int, cv_seeds=(MODEL_SEED,)) -> dict:
    """GIT_STRATEGY §9.2 형식. seeds 는 CV 분할 seed 목록이다."""
    return {"method": "StratifiedKFold", "n_splits": int(n_splits),
            "shuffle": True, "seeds": [int(s) for s in cv_seeds],
            "model_seed": int(model_seed)}


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
                   model_params=None, cv_seed=42, model_seed=MODEL_SEED,
                   n_splits=5, label=None, v=True, return_proba=False):
    """단일 CV 실행. fold 안에서 spec 을 다시 fit 한다.

    cv_seed 는 폴드 분할만, model_seed 는 모델 초기화만 지배한다 (팀 프로토콜).
    둘을 하나로 묶으면 '분할 변동'과 '모델 변동'이 섞여 σ 를 해석할 수 없다.
    """
    name, fn = MODELS[model_key]
    mp = model_params or DEFAULT_MODEL_PARAMS[model_key]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)
    oof = np.empty(len(y), dtype=object); dim = 0; t = time.time()
    classes_all = np.array(sorted(set(y)))
    proba = np.zeros((len(y), len(classes_all))) if return_proba else None
    for i_tr, i_va in cv.split(train, y):
        spec = fit_spec(train.iloc[i_tr], gene_cols, seed=model_seed)
        Xa, _ = build_features(train.iloc[i_tr], counts.iloc[i_tr], spec, blocks)
        Xb, _ = build_features(train.iloc[i_va], counts.iloc[i_va], spec, blocks)
        clf = fn(model_seed, mp).fit(Xa, y[i_tr])
        oof[i_va] = clf.predict(Xb); dim = Xa.shape[1]
        if return_proba:
            # fold 마다 등장 클래스가 다를 수 있으므로 전체 클래스 축에 맞춰 넣는다
            pos = [int(np.where(classes_all == c)[0][0]) for c in clf.classes_]
            proba[np.ix_(i_va, pos)] = clf.predict_proba(Xb)
    oof = np.array(list(oof))
    f1 = round(float(f1_score(y, oof, average="macro")), 5)
    acc = round(float(accuracy_score(y, oof)), 5)
    if v:
        print(f"         {(label or ''.join(blocks)):30} cv_seed {cv_seed}  dim {dim:5d}  "
              f"F1 {f1:.5f}  Acc {acc:.5f}  ({time.time() - t:.0f}s)", flush=True)
    out = {"blocks": "".join(blocks), "model": model_key, "model_name": name,
           "cv_seed": int(cv_seed), "model_seed": int(model_seed),
           "dim": int(dim), "f1_macro": f1, "accuracy": acc, "oof": oof}
    if return_proba:
        out["proba"] = proba
        out["classes"] = classes_all
    return out


def cross_validate_multi(train, y, counts, gene_cols, blocks, model_key="logreg",
                         model_params=None, cv_seeds=CONFIRMATION_CV_SEEDS,
                         model_seed=MODEL_SEED, n_splits=5, label=None, v=True):
    """여러 cv_seed 로 반복해 평균과 표준편차를 낸다.

    std 는 ddof=1 (팀 벤치마크와 동일). seed 가 1개면 std 는 None 이다.
    **주의** — n=3 의 표본표준편차는 그 자체로 불확실하다. σ 를 정밀한 상수처럼
    쓰지 말고 '이 정도 흔들린다'는 눈금으로만 쓴다.
    """
    runs = [cross_validate(train, y, counts, gene_cols, blocks, model_key, model_params,
                           cv_seed=s, model_seed=model_seed, n_splits=n_splits,
                           label=label, v=v) for s in cv_seeds]
    f1s = np.array([r["f1_macro"] for r in runs], dtype=float)
    accs = np.array([r["accuracy"] for r in runs], dtype=float)
    std = float(f1s.std(ddof=1)) if len(f1s) > 1 else None
    out = {
        "blocks": "".join(blocks), "model": model_key, "model_name": runs[0]["model_name"],
        "dim": runs[0]["dim"], "cv_seeds": [int(s) for s in cv_seeds],
        "model_seed": int(model_seed),
        "f1_macro": round(float(f1s.mean()), 5),
        "f1_macro_std": round(std, 5) if std is not None else None,
        "accuracy": round(float(accs.mean()), 5),
        "accuracy_std": round(float(accs.std(ddof=1)), 5) if len(accs) > 1 else None,
        "per_seed": [{"cv_seed": r["cv_seed"], "f1_macro": r["f1_macro"],
                      "accuracy": r["accuracy"]} for r in runs],
        "oof": runs[0]["oof"],          # 클래스별 분석용 (첫 seed 기준)
    }
    if v and len(f1s) > 1:
        print(f"         {'└ 평균':30} F1 {out['f1_macro']:.5f} ± {out['f1_macro_std']:.5f}  "
              f"Acc {out['accuracy']:.5f}", flush=True)
    return out


def run_pipeline(root=None, blocks=None, model=None, repeat=1, smoke=False,
                 submit_gate=None, v=True, write=True) -> dict:
    cfg = load_cfg()
    p = cfg["pipeline"]
    root = Path(root) if root else find_project_root()
    blocks = blocks or p["blocks"]
    model = model or p["model"]
    seed = p["cv"].get("model_seed", MODEL_SEED)
    cv_seeds = tuple(p["cv"].get("seeds") or [p["cv"].get("seed", MODEL_SEED)])
    if smoke:
        cv_seeds = cv_seeds[:2]
    n_splits = 2 if smoke else p["cv"]["n_splits"]
    dec = p["decimals"]
    gate = p["submit_gate"] if submit_gate is None else submit_gate
    ref = p["baseline"]
    # 모델 파라미터는 config.yaml 이 정본. 없으면 DEFAULT 로 떨어진다.
    mp = dict(p.get("model_params") or DEFAULT_MODEL_PARAMS[model])
    bt = tuple(blocks)
    name, fn = MODELS[model]
    exp_id = cfg["experiment"]["id"]

    _log("=" * 72, v)
    _log(f"  {exp_id} · pipeline {PIPELINE_VERSION} · features_A {fa.__version__}", v)
    _log(f"  피처 {blocks} · {name} · model_seed {seed} · "
         f"StratifiedKFold-{n_splits} · cv_seeds {list(cv_seeds)}", v)
    _log(f"  model_params {mp}", v)
    _log(f"  기준선 {ref['experiment']} = {ref['f1_macro']:.5f} (판정 {round(ref['f1_macro'], dec)})", v)
    _log("=" * 72, v)

    train, test, submission, gene_cols = load_data(root, smoke, v)
    y = train[TARGET].values
    ct, cte = parse_all(train, test, gene_cols, v)

    ok, checks = leakage_checks(train, test, cte, gene_cols, bt, seed, v)
    if not ok:
        raise RuntimeError("Leakage 자가검증 실패")

    _log(f"[Step 4] 교차검증 · cv_seeds {list(cv_seeds)}", v)
    r0 = cross_validate_multi(train, y, ct, gene_cols, bt, model, mp,
                              cv_seeds=cv_seeds, model_seed=seed,
                              n_splits=n_splits, v=v)

    # 결정성 — 같은 cv_seed 를 한 번 더 돌려 같은 값이 나오는지
    det = None
    if repeat > 1:
        again = cross_validate(train, y, ct, gene_cols, bt, model, mp,
                               cv_seed=cv_seeds[0], model_seed=seed,
                               n_splits=n_splits, label="결정성 재실행", v=v)
        det = again["f1_macro"] == r0["per_seed"][0]["f1_macro"]
        _log(f"         결정성 {'PASS' if det else 'FAIL'}", v)

    f1, acc, std = r0["f1_macro"], r0["accuracy"], r0["f1_macro_std"]
    gate_pass = round(f1, dec) > round(ref["f1_macro"], dec)

    _log("=" * 72, v)
    _log(f"  Macro F1 {f1:.5f}" + (f" ± {std:.5f}" if std is not None else "") +
         f"  ({f1 - ref['f1_macro']:+.5f})  Acc {acc:.5f}  "
         f"{verdict(f1, ref['f1_macro'], dec)}", v)
    if std is not None and std > 0:
        _log(f"  기준선 대비 {abs(f1 - ref['f1_macro']) / std:.1f}σ "
             f"(σ 는 cv_seed {len(cv_seeds)}개의 표본표준편차 — n 이 작아 그 자체로 부정확)", v)
    _log("=" * 72, v)

    res = {
        "experiment": exp_id, "owner": "member_d", "track": "A", "model": name,
        "seed": seed, "validation": validation_spec(n_splits, seed, cv_seeds),
        "model_parameters": mp, "accuracy": acc, "f1_macro": f1,
        "f1_macro_std": std, "accuracy_std": r0["accuracy_std"],
        "feature_blocks": blocks, "n_features_cv": r0["dim"],
        "baseline": ref, "delta_vs_baseline": round(f1 - ref["f1_macro"], 5),
        "verdict": verdict(f1, ref["f1_macro"], dec),
        "cv_runs": r0["per_seed"],
        "deterministic": bool(det) if det is not None else None,
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
            f"| CV Macro F1 (StratifiedKFold-{n_splits}, cv_seeds {list(cv_seeds)}) | "
            f"**{f1:.5f}**" + (f" ± {std:.5f}" if std is not None else "") + " |\n"
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
                               tuple(a.blocks or cfg["blocks"]),
                               cfg["cv"].get("model_seed", MODEL_SEED))
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
