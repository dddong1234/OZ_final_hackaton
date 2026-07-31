# SDH exp_003 — 전처리 10종 비교

공용 `run_preprocessing_benchmark()`를 사용해 모델과 5-Fold 조건을 고정하고
전처리만 비교한다. 1차 실험은 seed 42이며, 유망 후보만 `--confirmation`으로
seed 42/52/62 반복 검증한다.

| Case | 전처리 |
| --- | --- |
| 01 | 결측→WT, WT/변이 이진화 |
| 02 | 01 + fold train에서 상수 유전자 제거 |
| 03 | 02 + `log1p(변이 유전자 수)` |
| 04 | 02 + `log1p(실제 변이 토큰 수)` |
| 05 | 02 + 두 burden |
| 06 | 05 + synonymous/missense/nonsense/frameshift/other 개수 |
| 07 | 05 + fold train 변이 빈도 3 이상 유전자 |
| 08 | 05 + fold train 변이 빈도 5 이상 유전자 |
| 09 | 05 + fold train 변이 빈도 10 이상 유전자 |
| 10 | 05 + fold train 상위 50개 mutation token hotspot |

빈도 필터와 hotspot 목록은 각 fold의 train 부분에서만 학습한다. 실제 변이 토큰
수와 변이 유형은 WT 이진화 전에 원본 문자열에서 계산한다.

JupyterLab을 저장소 루트에서 열고 `experiment.ipynb`를 실행한다. 코드에서 직접
실행하려면 다음과 같이 사용한다.

```python
from experiments.SDH.exp_003_preprocessing.run_benchmark import run

leaderboard = run()
leaderboard[
    ["preprocessing", "oof_f1_macro_mean", "oof_accuracy_mean"]
]
```

빠르게 일부 case만 먼저 확인할 수도 있다.

```python
leaderboard = run(
    selected_cases=[
        "case_01_wt_binary",
        "case_03_gene_burden",
        "case_04_token_burden",
        "case_05_both_burdens",
    ]
)
```

1차 결과 상위 후보 반복 검증:

```python
leaderboard = run(
    selected_cases=[
        "case_06_mutation_types",
        "case_10_hotspot_top50",
        "case_09_min_count_10",
    ],
    confirmation=True,
)
```

일반 실행은 `metrics_<case>.json`과 `leaderboard.csv`, 반복 검증은
`metrics_<case>_confirmation.json`과 `leaderboard_confirmation.csv`로 분리된다.
`results/` 아래 실행 결과는 커밋하지 않는다.

LR에서 선별된 후보의 LightGBM 2차 검증은 `lightgbm_verification.ipynb`에서
실행한다. 비교 기준을 포함해 다음 네 후보만 실행한다.

```python
lgbm_leaderboard = run(
    selected_cases=[
        "case_01_wt_binary",
        "case_06_mutation_types",
        "case_10_hotspot_top50",
        "case_09_min_count_10",
    ],
    model="lightgbm",
)
```

결과는 `metrics_<case>_lightgbm.json`과 `leaderboard_lightgbm.csv`로 분리된다.

실험 결과와 결론은 `EXPERIMENT_SUMMARY.md`에 정리했다. LR과 LightGBM 모두
`case_06_mutation_types`가 1위였으며, exp_004에서는 이를 기준으로 빈도 필터와
hotspot 조합을 비교한다.
