# SDH exp_013 — standalone champion baseline

> **팀의 후속 모델 실험은 이 파이프라인을 공통 기준이자 시작점으로 사용한다.**
> 각 담당자는 데이터 분할·누수 방지·OOF 저장 계약을 유지하고, 담당 모델에
> 맞는 입력 표현만 독립적으로 변경한다.

## 실험 질문

외부 실험 모듈을 import하지 않는 독립 baseline에서 고정 도메인 피처를 제거하고,
fold-train 통계만으로 재현 가능한 공용 검증 기준을 제공할 수 있는가?

## 구현 범위

- mutation 문자열 정규화 및 mutation type 분류
- G/B/V/T/R/A/S B04 backbone
- fold-train support 5 이상 recurrent missense
- 고정 exact mutation 및 고정 암종쌍 contrast 제거
- gene×mutation-type class enrichment 26개
- support 10, shrinkage 10, clip 4
- inner 5-fold cross-fit 및 train-score 표준화
- Logistic Regression: lbfgs, C=0.07, max_iter=2000, balanced

`standalone_pipeline.py`는 다른 실험 모듈을 import하지 않는다. train과
validation/test 원본을 결합하지 않으며 vocabulary, support, 피처 선택,
표준화 통계는 모두 fit 분할에서만 계산한다. `use_fixed_contrast=True`를 넘기면
실행을 중단해 기존 고정 암종쌍 경로가 다시 활성화되지 않도록 한다.

## 검증 순서

1. synthetic train/test 불변성 검사
2. 고정 `C__`, `D__exact` 열 부재 assertion
3. seed 42 5-fold CV
4. seeds 42/52/62 확인
5. 최종 후보 잠금 후 full-train 제출 설계행렬 및 예측 생성

## 확정 결과

| 항목 | 결과 |
| --- | ---: |
| seed 42 OOF Macro F1 | 0.5261303511 |
| seed 52 OOF Macro F1 | 0.5292720822 |
| seed 62 OOF Macro F1 | 0.5274239869 |
| 3-seed 평균 | **0.5276088068** |
| CV 피처 수 | 평균 8,193.2 |
| 수렴 경고 | 0 |
| test 변경 불변성 | PASS |

exp14에서 같은 안전 피처를 LGBM specialist와 결합한 결과, LR 80% + LGBM 20%
3-seed 평균이 `0.539845`로 레거시 고정 암종쌍 버전 `0.538052`보다
`+0.001793` 높았다. 규정 위험 요소를 제거해도 앙상블 안정성이 유지·개선됐으므로
이 안전 standalone을 팀 공용 baseline으로 승격한다.

## 레거시 결과 참고

기존 `0.528236` 평균과 제출 예측 parity는 고정 exact mutation 및 고정 암종쌍을
포함한 재현 결과이므로 legacy로만 보관한다. 후속 모델·FE 실험은 안전 baseline
`0.527609`를 비교 기준으로 사용한다.

## 팀 후속 연구

모델별 입력 표현, 예측 다양성 평가, OOF/test 확률 저장 계약과 앙상블 채택
기준은 [MODEL_DIVERSITY_STRATEGY.md](MODEL_DIVERSITY_STRATEGY.md)를 따른다.
