# SDH exp_007 — 유망 FE 조합 탐색

## 목적

새로운 단일 피처를 더 발굴하기보다, exp_003~006과 팀 FE setting DB에서 유망했던
피처 블록을 같은 Logistic Regression 조건에서 조합한다.

## 고정 모델 조건

- `LogisticRegression`
- `solver="lbfgs"`
- `C=0.07`
- `max_iter=2000`
- `class_weight="balanced"`
- Stratified 5-fold
- 1차 seed 42, 상위 후보 확인 seed 42/52/62

## 후보

| Case | 구성 |
| --- | --- |
| 01 | mutation types + hotspot50 + burden2 기준 |
| 02 | case 01 + multi-mutated-gene burden |
| 03 | case 02에서 hotspot100 |
| 04 | case 02 + truncating gene flags |
| 05 | burden3 + mutation types + recurrent missense(min 5), hotspot 대체 |
| 06 | case 02 + hotspot과 겹치지 않는 recurrent missense |
| 07 | case 06 + truncating gene flags |
| 08 | case 02 + fold-train min gene count 10 |

hotspot, recurrent missense, retained gene, truncating gene 목록은 모두 각 CV fold의
학습 분할에서만 결정한다. validation에는 학습된 목록을 적용만 한다.

## 실행

`experiment.ipynb`를 위에서부터 한 셀씩 실행한다. 각 case가 별도 셀이라 중간
결과를 확인하거나 특정 case만 다시 실행할 수 있다. 전체 실행을 감싼 `run()`
래퍼는 두지 않았다.

## 결과

3-seed 확인에서 functional full 조합이 OOF Macro F1
`0.43189 ± 0.00325`로 reference의 `0.41113 ± 0.00134`보다 `+0.02076`
높았다. truncating-only 조합도 `0.42896 ± 0.00295`로 강한 개선을 보였다.

상세 결과와 해석은 [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md)에 기록했다.
