# 3-seed Selective-EB + 자동 Specialist 제출 파이프라인

## 핵심 결론

현재 팀의 기준 제출은 **3-seed bagging Public LB 0.48112**입니다. 같은 전처리와 모델 구조를 seed `42`, `777`, `2024`에서 각각 full-train 학습한 뒤, 세 모델의 test 확률을 동일하게 평균했습니다.

이 문서는 특정 이전 실험 파일을 참조하지 않아도 이해할 수 있도록, 실제 제출에 사용한 전처리·피처·모델 결합을 직접 설명합니다.

## 데이터 처리 원칙

- train에서만 vocabulary, recurrent event, 암종별 통계, 표준화, specialist 대상을 학습한다.
- test는 학습된 변환을 적용하고 확률을 예측하는 용도로만 사용한다.
- train/test를 결합하지 않는다.
- WT, 빈 문자열, NaN은 변이 event로 만들지 않는다.
- 암종명, 유전자명, exact mutation 목록을 규칙으로 고정하지 않는다.

따라서 test의 결측치가 mutation으로 세어지지 않으며, 제출 audit에서 `nan_as_mutation_count=0`, `leakage_check=True`를 확인한다.

## 1. 변이 문자열을 구조화 event로 파싱

각 유전자 셀의 단백질 변이 문자열을 event 단위로 분리한다. `WT`, 공백, NaN은 event 0개로 처리한다. 정상 event는 missense, synonymous, nonsense, frameshift, splice, in-frame indel, 기타 유형으로 분류한다.

한 셀에 여러 event가 있으면 중복 없이 각각 보존한다. 이 event는 이후 모든 피처의 공통 원재료다.

## 2. 환자별 구조화 피처

| 블록 | 실제 피처 | 의미 |
| --- | --- | --- |
| G | 유전자별 mutation binary | 어떤 유전자에 변이가 있었는가 |
| B | mutated gene 수, 전체 event 수, 복수 event 유전자 수 | 환자의 전체 변이량과 복잡도 |
| V | 기능성 event-type별 count | missense·nonsense·frameshift 등의 구성 |
| T | truncating event가 있는 유전자 및 총개수 | 기능 상실성 변이 구조 |
| R | fold-train에서 5회 이상 나온 recurrent missense exact event | 반복적으로 관찰된 구체적 변이 |
| A-pair | ref→alt 아미노산 치환 방향 380종 count의 `log1p` | 변이 방향의 누적 패턴 |
| S | event 수·유전자당 다중 event·event type 다양성·entropy·dominant share 등 8개 | 한 환자 내 event topology |

`R`의 후보와 모든 유전자/event vocabulary는 해당 학습 fold에서만 만든다. validation/test에만 나타난 event는 새로운 열을 만들지 않고 학습된 열에 투영만 한다.

## 3. gene×event-type 암종 enrichment

`TP53__MISSENSE`처럼 **유전자와 기능성 변이 유형의 조합**이 각 암종에서 다른 암종보다 얼마나 많이 나타나는지 계산한다. 각 암종마다 하나의 evidence score를 만들므로 26개 dense score가 생성된다.

점수는 outer train 내부 5-fold cross-fitting으로 train OOF를 먼저 만들고, 이 train OOF의 평균·표준편차로 표준화한다. validation/test에는 outer/full train으로 학습한 통계만 적용한다.

## 4. Empirical-Bayes enrichment와 selective gate

Empirical-Bayes(EB)는 희귀 `gene×event-type` token을 즉시 버리지 않고, 전체 train의 발생률 쪽으로 보수적으로 수축해 암종별 evidence를 계산한다.

- non-EB LR: 구조화 피처 + 일반 enrichment
- EB LR: 구조화 피처 + Empirical-Bayes 26개 암종 evidence

EB LR의 Top-1과 Top-2 확률 차이가 **0.05 미만**이면 non-EB LR 확률을, 그 외에는 EB LR 확률을 사용한다. 이 margin은 사전 검증 후 고정됐으며 제출에서 다시 탐색하지 않는다.

## 5. 자동 혼동쌍 LGBM specialist

다중분류 LGBM을 구조화 피처로 학습한다. 이어 학습 데이터의 암종별 mutation centroid cosine similarity로 유사 암종쌍 2개를 자동 발견한다. 각 쌍에는 binary LGBM specialist를 학습하지만, 전체 26개 클래스 확률을 새로 만들지는 않는다. 원래 LGBM이 그 쌍에 배정한 확률 질량만 두 암종 사이에서 재배분하고 다른 클래스 확률은 보존한다.

최종 확률은 다음과 같다.

```text
0.80 × selective-EB LR probability
+ 0.20 × automatic-specialist LGBM probability
```

## 6. 3-seed bagging 제출

최종 파이프라인을 full train으로 seed `42`, `777`, `2024`에서 각각 학습하고, 각 test 행·클래스 확률을 `1/3`씩 평균해 가장 큰 확률의 암종을 제출한다.

3-seed OOF 검증은 모델의 안정성 확인이고, 3-seed bagging은 test 예측 분산을 줄이기 위한 최종 추론 방식이다.

## 검증 근거와 현재 기준

Selective-EB branch는 기존 H0 대비 3-seed OOF에서 평균 `+0.004669`, 상승 fold `12/15`를 기록했다. 검증을 통과한 동일 seed만 bagging에 사용했고, 현재 bagging 제출의 Public LB는 **0.48112**다.

주의: 서로 다른 CV fold에서 나온 OOF 확률을 사후 평균해 산출한 점수는 올바른 bagging CV가 아니므로, 성능 근거로 사용하지 않는다.

## 실행과 산출물

다른 사람이 실행할 때는 다음 standalone 파일 하나만 실행하면 된다.

```bash
python experiments/gs/notebooks/submission/reproduce_h0_selective_eb_3seed_standalone.py
```

필요한 데이터는 `data/raw/train.csv`, `test.csv`, `sample_submission.csv`다. 실행 후 제출 CSV와 동일 이름의 audit JSON이 `experiments/gs/notebooks/submission/`에 생성된다.
