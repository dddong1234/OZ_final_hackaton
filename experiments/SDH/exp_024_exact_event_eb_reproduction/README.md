# SDH exp_024 — Exact-event EB 독립 재현

## 한 줄 요약

기존 `gene × event-type` 암종별 증거를 유지하면서, `TP53__R175H`처럼 정규화된 정확 변이 사건별 암종 증거 26개를 추가한 모델을 **test와 LB 없이** 독립 검증한다.

## 실험 질문

`TP53__MISSENSE`처럼 서로 다른 변이를 하나로 묶는 표현보다 exact event를 Empirical-Bayes class score로 압축한 표현이 일반화 성능을 높이는가?

## 제공 결과

| 항목 | 값 |
| --- | ---: |
| 3-seed OOF Macro F1 | 0.568441 ± 0.002310 |
| H0 대비 평균 OOF 변화 | +0.021186 |
| Public LB | 0.5086 |
| seeds | 42 / 777 / 2024 |

위 값은 제공 문서의 기준값이다. 이 폴더에서 직접 실행한 결과는 실행 후 별도로 기록한다.

## 핵심 구조

1. mutation 문자열을 행 단위로 파싱한다.
2. 구조화 H0 피처와 기존 gene×event-type EB score를 생성한다.
3. fold-train에서 관측된 모든 비상수 exact event에 암종별 posterior log-odds를 학습한다.
4. 희귀 event는 `shrinkage=20`으로 전역 prior에 수축한다.
5. train용 supervised score는 inner 5-fold OOF로 만들고, validation/test에는 train fit을 적용만 한다.
6. exact-event EB LR의 margin이 0.05 미만이면 non-EB LR로 되돌린다.
7. selective LR 0.8과 automatic LGBM specialist 0.2를 결합한다.
8. seeds 42/777/2024의 test 확률을 동일 가중 평균한다.

## 누수 계약

- raw train/test concat 금지
- vocabulary, support, EB weight, scaling은 train에서만 학습
- supervised train feature는 inner cross-fit으로 생성
- 고정 암종·유전자·exact mutation 목록 없음
- test는 최종 transform/predict 단계에서만 읽음

## 검증 설계

- 비교 기준: 동일 fold의 H0 `0.8 × non-EB LR + 0.2 × specialist`
- 후보: `0.8 × selective Exact-event EB LR + 0.2 × specialist`
- outer CV: Stratified 5-fold
- seeds: 42 / 777 / 2024
- inner CV: gene×type EB와 exact-event EB train feature를 위한 5-fold cross-fit
- 추가 분석: seed별, fold별, class별 F1와 H0 오답 복구/신규 오류
- 선택 감사: permutation-label seed42 5-fold

### 채택 기준

- 3-seed 평균 개선 `≥ +0.010`
- 최소 seed 개선 `≥ +0.005`
- 3개 seed 모두 개선
- 15개 fold 중 11개 이상 개선
- permutation label에서 exact-event EB의 gene×type EB 대비 개선 `< +0.010`

## 독립 재현 결과

| 항목 | 결과 |
| --- | ---: |
| H0 3-seed 평균 OOF Macro F1 | 0.542836 |
| Exact-event 최종 3-seed 평균 | **0.567999 ± 0.003094** |
| H0 대비 평균 개선 | **+0.025163** |
| 최소 seed 개선 | **+0.024756** |
| 상승 seed | **3/3** |
| 상승 outer fold | **15/15** |
| 제공 결과 0.568441과 차이 | **-0.000442** |
| 수렴 경고 | 0 |

모든 사전 채택 기준을 통과했다. 클래스별 평균 개선 상위는 DLBC
`+0.1331`, LGG `+0.1105`, GBMLGG `+0.0937`, KIPAN `+0.0690`, KIRC
`+0.0667`이었다. Exact-event 전용 permutation-label 검사는 specialist의
겹치는 자동 암종쌍 문제를 분리한 LR-only 감사 코드로 수정했으며 결과는 아직
미기록 상태다.

## 실행 방법

저장소 루트에서 JupyterLab을 시작한 뒤 `experiment.ipynb`를 위에서 아래로 실행한다. seed별 학습을 별도 셀로 분리했으므로 진행 상황을 확인하면서 실행할 수 있다. **노트북은 test.csv를 읽지 않으며 제출 파일을 만들지 않는다.**

생성되는 경량 fold/class/metric/audit 파일은 `results/`에 저장된다. OOF 확률과 모델은 저장하거나 커밋하지 않는다.

## 파일

- `exact_event_pipeline.py`: 제공된 standalone 파이프라인의 저장소 내 사본
- `oof_validation.py`: H0와 Exact-event EB의 fold-local 비교 및 permutation 검사
- `experiment.ipynb`: 사용자 실행용 3-seed OOF 워크플로
- `FIXED_CONSTANTS.md`: 제공된 고정 상수와 안전성 계약

## 주의

`exact_event_pipeline.py`는 제공된 standalone의 원문 사본이다. `oof_validation.py`가 그 안의 고정 함수와 상수를 사용하되 outer validation까지 fold-local하게 감싸 제공 OOF 결과를 독립 재검증한다.

