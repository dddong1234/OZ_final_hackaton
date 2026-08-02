# exp-gs-002-09 설계 — LGBM 결측 노이즈 augmentation

## 목표

임의 결측 노이즈를 fold-train에만 주입했을 때 LightGBM의 원본 검증 데이터 OOF Macro F1이 유지 또는 개선되는지 확인한다.

## 범위와 비교

- 후보 A: 원본 train으로 학습하는 LGBM 기준선
- 후보 B: 동일 조건에서 fold-train에만 결측 노이즈를 주입하는 LGBM
- 모델·입력 피처·CV split·평가 방식은 두 후보에서 동일하게 유지한다.

## 입력과 모델

- 입력은 원본 유전자 G 피처만 사용한다.
- 관측 WT는 0, 하나 이상의 변이 문자열은 1로 인코딩한다.
- train 데이터에는 원래 결측이 없음을 검증한다.
- 모델은 LightGBM multiclass이며, 모델 파라미터는 두 후보 간 동일하게 고정한다.

## 고정 마스킹 규칙

- `MASK_RATE = 0.001`
- `MASK_SEED = 42`
- 각 fold의 train 부분에서만 전체 유전자 셀을 균일 무작위로 선택해 `NaN`으로 바꾼다.
- validation fold는 원본 상태로 유지한다.
- 후보 B는 마스킹된 train으로만 fit하며, 후보 A에는 마스킹을 적용하지 않는다.

## 누수 방지

- 이 노트북은 `train.csv`만 읽고 `test.csv`를 읽지 않는다.
- test의 결측 개수·위치·분포·통계를 마스킹률, seed, 대상 선정, 모델 파라미터에 사용하지 않는다.
- 피처 인코딩과 모델 학습은 각 fold의 train에서만 수행한다.
- 결과에는 `test_read=False`, train 원본 결측 수, 마스킹 rate/seed/셀 수를 기록한다.

## 검증과 판정

- Stratified 5-fold, CV seed 42로 후보 A와 B를 비교한다.
- 지표는 OOF Macro F1이며 OOF, 클래스별 F1, 실행 메타데이터 JSON, 요약 CSV, 시각화를 저장한다.
- 후보 B가 후보 A보다 개선될 때에만 42/2024/777 3-seed 안정성 검증 후보로 넘긴다. 그렇지 않으면 09에서 기각한다.
- 이 실험은 결측이 있는 validation/test 성능을 직접 재현하지 않는다. 결측 노이즈를 포함한 학습이 원본 OOF에 미치는 영향만 평가한다.
