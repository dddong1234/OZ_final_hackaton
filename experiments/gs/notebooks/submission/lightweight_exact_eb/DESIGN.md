# Lightweight Exact-event EB 운영 모델 설계

## 목적

대회용 3-seed LR/LGBM 앙상블과 별도로, 동일한 핵심 변이 표현을 사용하는 단일 Logistic Regression 운영 모델을 제공한다. 목표는 빠른 재학습·재현 가능한 CV·실시간 추론 연결이다.

## 입력과 안전 계약

- 학습: `train.csv`만으로 parser, vocabulary, EB 통계, 표준화, LR을 fit한다.
- 추론: 저장된 train-fitted 상태만 사용하여 새 환자 또는 test를 transform/predict한다.
- WT·빈 문자열·NaN은 event를 만들지 않는다.
- 고정 암종·유전자·exact mutation 목록은 사용하지 않는다.
- test를 vocabulary, 통계, scaling, feature selection에 사용하지 않는다.

## 모델

- 구조화 mutation feature: mutation binary, burden, event-type count, truncation, recurrent missense, A-pair, topology.
- gene×event-type Empirical-Bayes 26 class score.
- gene×exact-event Empirical-Bayes 26 class score.
- 최종 분류기: `LogisticRegression(lbfgs, C=0.07, max_iter=2000, class_weight='balanced')` 단일 모델.
- 제거: LGBM specialist, selective gate, 3-seed bagging.

## 생성 파일

- `lightweight_exact_eb_core.py`: 학습/저장/로딩/단일 및 배치 추론 공통 로직.
- `train_lightweight_exact_eb.py`: full-train 모델 번들 저장.
- `evaluate_lightweight_exact_eb_cv.py`: 5-fold × 3-seed OOF Macro F1 및 안전 감사.
- `predict_lightweight_exact_eb.py`: 제품 연동용 JSON/CSV 배치 추론.
- `generate_lightweight_exact_eb_submission.py`: 모델 번들로 제출 CSV 생성.
- `README.md`: 실행 순서와 API 예시.

## 검증

- parser NaN/WT 단위 테스트
- train-only vocabulary projection 테스트
- 저장/로딩 후 확률 일치 테스트
- 작은 subset smoke test
- 문법/import 검사

전체 3-seed CV 및 full-train 제출 생성은 사용자가 실행한다.
