# SDH exp_010 — 챔피언 FE 단계별 ablation

챔피언 보고서와 exp009의 차이를 한 번에 복사하지 않고, 전처리 블록을 하나씩
누적해 확인한다. Logistic Regression 파라미터와 CV 프로토콜은 고정한다.

## 고정 조건

- Logistic Regression `solver="lbfgs"`, `C=0.07`, `max_iter=2000`
- `class_weight="balanced"`
- Stratified 5-fold
- 1차 seed 42, leaderboard 상위 3개를 seed 42/52/62로 확인
- 각 fold train에서만 recurrent, contrast, exact top-4 후보를 학습
- exact hotspot 이름은 외부에서 하드코딩하지 않고 train-only 빈도·집중도로 선택

## 누적 후보

| Case | 추가 내용 |
| --- | --- |
| 01 | exp009 최고 FE: functional full + A pair raw count |
| 02 | 01 + A pair `log1p` |
| 03 | 02 + S topology/distribution |
| 04 | 03 + train-only KIRC↔KIPAN, LGG↔GBMLGG contrast |
| 05 | 04 + train-only exact mutation top-4 |

챔피언 보고서의 고정 `BRAF V600E` 등 4개 이름은 외부 지식 사용 가능성이 있어
그대로 사용하지 않는다. Case 05는 동일한 목적을 train-only 후보 선택으로 대체한
안전성 확인용이다.

## 실행

`experiment.ipynb`를 위에서부터 실행한다. 각 case는 별도 셀로 분리되어 있으며,
실제 CV 실행은 사용자가 직접 한다. 결과는 `results/` 아래에 저장되고 Git에는
커밋하지 않는다.

## 최종 결과

`case_02_plus_pair_log1p`가 seed 42에서 0.48208, seed 42/52/62에서
**0.48248 ± 0.00098**로 가장 높았다. A pair raw 대비 +0.01923 개선이다. 이후
S, train-only contrast, train-only exact top-4를 누적하면 점수가 차례로 하락해
채택하지 않았다. 세부 결과와 B04 비교 시 주의점은 `EXPERIMENT_SUMMARY.md`에
기록했다.
