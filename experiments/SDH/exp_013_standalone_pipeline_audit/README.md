# SDH exp_013 — standalone champion baseline

> **팀의 후속 모델 실험은 이 파이프라인을 공통 기준이자 시작점으로 사용한다.**
> 각 담당자는 데이터 분할·누수 방지·OOF 저장 계약을 유지하고, 담당 모델에
> 맞는 입력 표현만 독립적으로 변경한다.

## 실험 질문

외부 실험 모듈을 import하지 않고 exp12 챔피언 전처리를 직접 구현했을 때
기존 점수와 예측을 재현하면서 train/test 및 fold 누수 검사를 더 명확하게
통과할 수 있는가?

## 구현 범위

- mutation 문자열 정규화 및 mutation type 분류
- G/B/V/T/R/A/S B04 backbone
- exact mutation 4개
- 고정 contrast pair 2개(기존 결과 parity 전용 옵션)
- gene×mutation-type class enrichment 26개
- support 10, shrinkage 10, clip 4
- inner 5-fold cross-fit 및 train-score 표준화
- Logistic Regression: lbfgs, C=0.07, max_iter=2000, balanced

`standalone_pipeline.py`는 다른 실험 모듈을 import하지 않는다. train과
validation/test 원본을 결합하지 않으며 vocabulary, support, 피처 선택,
contrast, 표준화 통계는 모두 fit 분할에서만 계산한다.

## 검증 순서

1. synthetic train/test 불변성 검사
2. seed 42 5-fold CV
3. 기존 exp12 seed 42 결과와 점수 비교
4. full-train 제출 설계행렬 및 예측 생성
5. 기존 exp12 제출 예측 2,546행과 완전 동일성 검사
6. 통과 후 3-seed 확인

고정 contrast는 기존 exp12 재현을 위한 별도 옵션이다. 규칙 보수형 후보는
`use_fixed_contrast=False`로 추가 비교한다.

## 확정 결과

| 항목 | 결과 |
| --- | ---: |
| seed 42 OOF Macro F1 | 0.5291849039 |
| seed 52 OOF Macro F1 | 0.5299051698 |
| seed 62 OOF Macro F1 | 0.5256170622 |
| 3-seed 평균 | **0.5282357120** |
| full-train 피처 수 | 8,425 |
| exp12 제출 대비 변경 예측 | **0 / 2,546행** |
| 수렴 경고 | 0 |
| test 변경 불변성 | PASS |

standalone 구현은 기존 exp12의 seed별 점수와 제출 예측을 완전히 재현했다.
원본 train/test 결합 없이 동일 결과를 얻었으므로 앞으로는 외부 실험 모듈을
연쇄 참조하는 기존 코드보다 이 구현을 기준으로 사용한다.

## 고정 contrast 참고

고정 `KIRC↔KIPAN`, `LGG↔GBMLGG` contrast를 제거하면 seed 42 Macro F1이
`0.529185 → 0.526792`로 `-0.002393` 하락했다. 기존 결과 parity에는 고정
contrast를 사용하고, 규칙 보수형 연구에서는 fold-train 혼동행렬 기반 자동
발견 방식과 별도로 비교한다.

## 팀 후속 연구

모델별 입력 표현, 예측 다양성 평가, OOF/test 확률 저장 계약과 앙상블 채택
기준은 [MODEL_DIVERSITY_STRATEGY.md](MODEL_DIVERSITY_STRATEGY.md)를 따른다.
