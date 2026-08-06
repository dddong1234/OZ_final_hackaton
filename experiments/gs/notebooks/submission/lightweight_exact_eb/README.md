# Lightweight Exact-event EB

실시간/배치 운영을 위한 단일 Logistic Regression 모델입니다. 대회용 3-seed bagging과 LGBM specialist는 사용하지 않지만, train-only 구조화 변이 피처와 gene×event-type·exact-event Empirical-Bayes 26차원 점수는 유지합니다.

## 1. 학습 및 모델 저장

```bash
.venv/bin/python train_lightweight_exact_eb.py \
  --train-csv data/raw/train.csv \
  --bundle-out artifacts/lightweight_exact_eb_seed42.joblib \
  --seed 42
```

`joblib` bundle에는 parser 규칙, train-only vocabulary, EB weights/표준화값, feature 순서, class 순서, LR 모델이 함께 저장됩니다.

## 2. 제품 배치 추론

```bash
.venv/bin/python predict_lightweight_exact_eb.py \
  --bundle artifacts/lightweight_exact_eb_seed42.joblib \
  --input-csv new_patients.csv \
  --output-csv predictions.csv
```

입력은 `ID`와 학습 때의 유전자 열 순서를 가져야 합니다. 출력은 `ID`, 예측 `SUBCLASS`, 암종별 확률 열입니다. 추론은 저장된 bundle을 적용만 하며 재학습하지 않습니다.

## 3. CV 점수 확인

```bash
.venv/bin/python evaluate_lightweight_exact_eb_cv.py \
  --train-csv data/raw/train.csv \
  --result-dir results/lightweight_exact_eb_cv
```

5-fold × seeds `42/777/2024` OOF Macro F1, fold metrics, OOF predictions, leakage audit JSON을 저장합니다. CV 과정에서는 test를 읽지 않습니다.

## 4. 제출 CSV 생성

```bash
.venv/bin/python generate_lightweight_exact_eb_submission.py \
  --bundle artifacts/lightweight_exact_eb_seed42.joblib \
  --test-csv data/raw/test.csv \
  --sample-submission data/raw/sample_submission.csv \
  --output-csv submission_lightweight_exact_eb.csv
```

제출 CSV는 `ID`, `SUBCLASS`만 유지하며 sample submission의 ID 순서를 검증합니다.

## 안전 계약

- train-only vocabulary·EB 통계·scaling·model fitting
- test/new patient는 저장된 변환을 적용하는 추론 전용
- WT, 빈 값, NaN은 mutation event가 아님
- 고정 암종명·유전자명·exact mutation 목록 없음
