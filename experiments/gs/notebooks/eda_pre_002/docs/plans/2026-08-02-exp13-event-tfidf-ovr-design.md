# exp-gs-002-13 — event-token TF-IDF OVR 설계

## 목표

12번의 event-token TF-IDF 전처리는 고정한 채, 다중 클래스 Logistic Regression의 결정 구조만 multinomial에서 One-vs-Rest(OVR)로 바꿨을 때 08 primary 앙상블의 보완성이 커지는지 확인한다.

## 고정 조건

- 후보: `H-AS-LR-exact-confusion-pairs-Apair-log1p`
- TF-IDF token: `G__`, `E__`, `TYPE__`, `AA__`; `min_df=3`, unigram, `sublinear_tf=True`, `norm='l2'`
- LR: `lbfgs`, `C=0.07`, `max_iter=2000`, `class_weight='balanced'`
- 검증: Stratified 5-fold, CV seeds `42 / 2024 / 777`
- OOF에서는 train.csv만 읽고, fold-train으로만 vocabulary·IDF·모델을 fit한다.
- WT/NaN은 token을 만들지 않으며 `leakage_check=True`, `nan_as_mutation_count=0`을 검증한다.

## 비교 대상

동일 outer fold에서 primary, multinomial token LR, OVR token LR을 모두 학습한다.

1. 08 primary LR
2. event-token multinomial LR
3. event-token OVR LR
4. primary + multinomial token LR 확률 0.5/0.5 평균
5. primary + OVR token LR 확률 0.5/0.5 평균

각 seed에서 Macro F1, Accuracy, 클래스별 F1, vocabulary 크기, 수렴 경고를 저장한다. token 다항/OVR의 argmax 예측 불일치율도 저장한다.

## 판정

- 채택 후보: OVR 0.5 앙상블 평균 Macro F1이 12번 다항 0.5 앙상블(`0.480305`) 이상이고, 큰 단일-seed 하락·수렴 경고·안전성 위반이 없을 때.
- 그 외: 미검출. `min_df`, blend 비율, 파라미터는 이 결과로 재탐색하지 않는다.
