# H0 Prediction Stability Audit Implementation Plan

**Goal:** Train-only, fold-aligned H0 OOF probabilities로 seed 민감도와 bagging의 실제 오류 회복을 감사한다.

**Architecture:** `exp_model_010`의 유효한 fold-aligned OOF CSV를 입력으로만 사용한다. 새 모델·피처·임계값은 만들지 않고, 세 seed와 bagged 확률의 행별 예측 일치도·확률 분산·오류 전이를 계산한다. 모든 산출물은 `exp_model_014/result`에 저장한다.

**Constraints:** test 미열람, train/test 결합 없음, 새 학습 없음, 고정 암종/유전자/변이 규칙 없음, 결과는 다음 모델 설계의 감사 근거로만 사용한다.

## Files

- `common/run_h0_prediction_stability_audit.py`: CSV 읽기, 무결성 검사, 감사 지표와 그래프 생성.
- `common/test_h0_prediction_stability_audit.py`: 합성 OOF 입력으로 schema와 오류 전이 단위 테스트.
- `exp/exp-h0-prediction-stability-audit-01.ipynb`: 실행, 결과 표·그래프, 자동 요약.
- `docs/2026-08-05-h0-prediction-stability-audit.md`: 실험 목적·판정 해석.

## Validation

1. `py_compile`로 문법 검사.
2. 합성 확률 CSV로 unit test 실행.
3. `--smoke`로 실제 입력 경로/스키마와 확률 합을 읽기 전용 검사.
4. full audit는 사용자가 실행한다. 실행기는 test.csv를 열지 않는다.
