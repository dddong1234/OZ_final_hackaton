# Class-conditional Evidence Set Network Design

## 목표

팀 3-way ensemble Public LB `0.45348`을 중간 기준선으로 동결한다. 새 축은 P1+EB가 만든 암종별 event 증거를 단순 합산하지 않고, 환자별·암종 후보별 event evidence set으로 유지해 26개 후보를 직접 순위화한다.

## 범위와 중단 규칙

- 먼저 train-only raw vs normalized profile purity 감사를 수행한다.
- 다음으로 seed 42에서 Evidence Set Network를 screen한다.
- team 3-way ensemble OOF 대비 `+0.030`, 5 fold 중 4개 상승, low-margin Macro F1 `+0.040`, 15개 이상 클래스 개선을 모두 충족할 때만 3-seed로 확장한다.
- 위 조건을 만족하지 않으면 학습률, hidden size, pooling, threshold, blend를 재탐색하지 않고 축을 종료한다.
- team 3-way ensemble은 결과와 무관하게 제출 기준선으로 보존한다.

## 비교 기준 재현 계약

현재 저장소에는 Public LB `0.45348`을 만든 최종 3-way ensemble의 행 정렬된 OOF 확률이 보관되어 있지 않다. 따라서 Evidence Set screen은 과거 요약 점수만 읽어 비교하지 않는다. `exp_model_005/common` 안에 팀이 제공한 최종 파이프라인의 train-only 등가 구현을 둬 같은 train 행 순서·class order·seed42 outer fold에서 OOF 확률을 다시 만든다.

- vocabulary는 train fold에서만 생성하고 validation에는 projection만 적용한다. train/validation cache를 결합해 factorize하지 않는다.
- multinomial LR / OVR LR / LightGBM의 고정 비율은 `0.55 / 0.30 / 0.15`이며, 이 screen에서 비율이나 각 모델 파라미터를 탐색하지 않는다.
- 재현 OOF가 팀 기준 `0.54202`에서 ±`0.003`를 벗어나면 Evidence Set의 점수는 참고용으로만 저장하고 승격 판정을 내리지 않는다.
- 이 baseline 재현은 새로운 제출 파일을 만들지 않으며, test 파일도 읽지 않는다.

## 입력 계약

모든 문자열은 제공된 train cell에서만 파싱한다. WT, 빈 문자열, NaN은 event를 만들지 않는다. 외부 유전자 annotation, pathway, 단백질 서열, 외부 환자 데이터, test-derived vocabulary/statistics는 사용하지 않는다.

각 active event와 후보 암종의 입력은 다음만 사용한다.

- fold-train EB log-odds contribution
- token support와 EB posterior reliability
- 양/음 contribution flag
- canonical event-type one-hot
- 행 mutation burden으로 정규화한 contribution
- exact event, recurrent event 여부

event identity를 별도 외부 embedding으로 학습하지 않는다. gene×event-type EB weight는 outer train 내부의 inner OOF로 training sample에 생성하고, outer validation에는 outer-train 전체 weight만 적용한다.

## 모델

입력 tensor는 `[batch, 26 candidates, variable events, evidence dimensions]`이다. 작은 shared event MLP가 event evidence를 변환하고 masked mean/max/sum pooling으로 후보 벡터를 만든다. 후보 score를 26개 동시에 산출하고 환자별 listwise softmax + class-balanced cross entropy를 사용한다. 26개 독립 binary model을 만들지 않는다.

고정 screen 설정은 PyTorch CPU/MPS 호환, hidden 32, dropout 0.15, AdamW learning rate `0.001`, weight decay `0.0001`, batch 64, epoch 60, model seed 42다. early stopping과 hyperparameter 탐색은 사용하지 않는다.

## 검증과 산출물

- seed 42 outer/inner Stratified 5-fold
- team 3-way ensemble은 동일 fold·동일 class order 기준으로 재현하거나, 재현 검증된 OOF artifact만 같은 행 순서에서 사용한다.
- 저장: summary, fold/class/low-margin metrics, profile purity audit, OOF probabilities, feature contract, runtime, leakage audit JSON.
- 감사: test read false, train-only vocabulary, inner OOF EB for outer-train, outer validation label not used for EB/model fit, NaN-as-mutation 0, finite loss, class order equality.

## raw profile 감사 해석

raw profile은 gene별 원문 cell을 그대로 유지한 profile이고, normalized profile은 대소문자·`p.` 접두사·구분자·중복을 정규화한 profile이다. label purity는 profile group별 다수 label 비율을 해당 group sample 수로 가중 평균한다.

raw purity가 normalized purity보다 유의미하게 높아도 이를 곧바로 feature로 쓰지 않는다. 이는 검사/입력 formatting artifact일 수 있으므로, 해당 차이는 별도 일반화 위험으로 기록한다.
