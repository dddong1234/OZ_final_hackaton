# iljun-logreg-002 — 결과 요약

| 지표 | 값 |
|---|---|
| CV Macro F1 (StratifiedKFold-5, cv_seeds [42, 52, 62]) | **0.37806** ± 0.00555 |
| CV Accuracy | 0.37252 |
| 기준선 member-d-logreg-001 | 0.36305 |
| 기준선 대비 | +0.01501 · [+] 향상 |
| 제출 게이트 | 통과 (기준 0.363) |

**지문** — pipeline `9856c8874893` · features `d164d6e17c32` · preprocess `2fa768957eeb`

> 상세(비커밋): `metrics_cv.json` · 팀 holdout: `metrics.json` (training.run)
> 재현: `python3 experiments/member_d/exp_002_variant_type/pipeline.py`
