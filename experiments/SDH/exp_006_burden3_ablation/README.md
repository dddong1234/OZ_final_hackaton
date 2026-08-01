# SDH exp_006 — burden 3종 ablation

공용 전처리 벤치마크의 모델 설정을 그대로 유지하고, 현재 최고 FE 조합에 세 번째
burden 한 개를 추가했을 때의 순수 증분만 측정한다.

## 비교

| Case | 구성 |
| --- | --- |
| 01 | gene binary + gene/token burden + mutation types + hotspot 50 |
| 02 | case 01 + multi-mutated-gene burden |

`multi-mutated-gene burden`은 한 환자에서 mutation token이 2개 이상 기록된 유전자
수를 세어 `log1p`한 값이다.

## 고정 조건

- 공용 `run_preprocessing_benchmark()`
- Logistic Regression 기본 `C=1.0`
- `class_weight="balanced"`
- Stratified 5-fold
- 1차 seed 42, 확인 seed 42/52/62
- hotspot은 각 fold train에서만 선택

## 실행

JupyterLab에서 `experiment.ipynb`를 위에서부터 실행한다. 명령행 재현이 필요하면:

```powershell
python -m experiments.SDH.exp_006_burden3_ablation.run_benchmark
python -m experiments.SDH.exp_006_burden3_ablation.run_benchmark --confirmation
```

## 결과

burden 3종은 3-seed OOF Macro F1 `0.38258 ± 0.00218`로 burden 2종의
`0.37770 ± 0.00704`보다 평균 `+0.00488` 높았다. 다만 seed 42에서는
`-0.00078`이므로 확정 채택하지 않고, 다음 공통 LR 설정에서 재검증한다.
