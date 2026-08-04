# SDH exp_014 — focal LGBM and train-discovered specialists

고정 도메인 피처를 제거한 규정 안전 버전의 재실행 결과를 기록한다.

## 실험 질문

고정 exact mutation과 고정 암종쌍을 제거한 exp13 기반 피처에서 focal-loss
LGBM과 fold-train이 자동 발견한 암종쌍 specialist가 도움이 되는가?

## 공통 조건

- 전처리: exp13 기반 피처에서 고정 `C__`, `D__exact` 제거
- recurrent missense `R__`는 outer-fold train support로 자동 발견
- 원본 train/test 결합 없음
- 모든 학습형 전처리: outer fold-train only
- 5-fold, seed 42 screen
- 채택 후보만 seeds 42/52/62 확인
- 주 지표: OOF Macro F1
- LR 기준: exp13 standalone LR

## 1단계 — 메인 LGBM

| Case | Loss | Class weight |
| --- | --- | --- |
| main_01 | multiclass | balanced |
| main_02 | multiclass | 없음 |
| main_03 | focal γ=1, α=0.25 | 없음 |
| main_04 | focal γ=2, α=0.25 | 없음 |
| main_05 | focal γ=1, α=0.25 | balanced |

공통 LGBM은 전달받은 기준값인 400 trees, learning rate 0.05, leaves 25,
min child samples 10을 사용한다. focal Hessian은 LightGBM 안정성을 위해 양수
diagonal approximation을 사용한다.

## 2단계 — train-discovered pair specialist

seed 42 메인 승자를 고른 후 각 outer-fold train의 클래스별 유전자 변이율
벡터를 계산한다. cosine similarity가 높은 암종쌍 두 개를 자동으로 선택하고,
binary LGBM specialist를 해당 fold-train 행만으로 학습한다.

- 모든 자동 선택 쌍: 100 trees, lr 0.02, leaves 20, min child 10
- 유사도 1위/2위 쌍별 단독, 두 쌍 soft mass α=0.15/0.30/0.50
- predicted-only soft 0.30 및 hard routing

암종 이름은 코드에 고정하지 않는다. 쌍 발견, 유전자 통계, specialist 학습에는
해당 outer validation 및 test의 행·label·통계를 사용하지 않는다.

## 3단계 — LR 다양성과 blend

모든 메인 및 specialist 후보에 대해 disagreement, probability correlation,
LR 오답 복구율, 역손실률, double fault, oracle Macro F1을 계산한다.

최고 후보는 exp13 LR과 `95/5`, `90/10`, `85/15`, `80/20`, `70/30` 고정
weight로 비교한다. seed 42 승자 weight를 잠근 뒤 3seed에서 확인한다.

## 실행

`experiment.ipynb`를 위에서 아래로 한 셀씩 실행한다. 전처리 fold cache를 먼저
한 번 만들고 메인 case들이 재사용하므로 같은 seed에서 전처리를 반복하지 않는다.

각 case의 연구 질문, 변경점, 기대 효과와 판정 방법은
[EXPERIMENT_CASES.md](EXPERIMENT_CASES.md)에 상세히 기록했다.

## 확정 결과

### 메인 LGBM seed 42

| Case | OOF Macro F1 |
| --- | ---: |
| multiclass balanced | **0.476313** |
| multiclass unweighted | 0.471995 |
| focal γ=1 | 0.470847 |
| focal γ=1 + balanced | 0.470847 |
| focal γ=2 | 0.464668 |

Focal loss는 채택하지 않고 일반 multiclass balanced를 메인으로 선택했다.

### Specialist seed 42

fold-train에서 자동 발견한 두 암종쌍의 predicted-only hard routing이 메인
LGBM을 `0.476313 → 0.492332`로 `+0.016020` 개선해 specialist 승자로 선택됐다.

### LR 80% + LGBM 계열 20%

| Seed | exp13 LR | LGBM + hard specialist | Blend | LR 대비 |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0.526130 | 0.492332 | **0.543679** | +0.017549 |
| 52 | 0.529272 | 0.488738 | **0.540053** | +0.010780 |
| 62 | 0.527424 | 0.505154 | **0.535802** | +0.008378 |
| 평균 | 0.527609 | 0.495408 | **0.539845** | **+0.012236** |

3개 seed 모두에서 개선됐다. 신규 안전 버전의 blend 평균은 레거시 고정 암종쌍
버전 `0.538052`보다 `+0.001793` 높은 `0.539845`다. LGBM 단독 성능은 LR보다
낮지만 오류 패턴이 달라 앙상블 보조 모델로 가치가 있다.

### Public LB 제출 결과

3-seed full-train 확률 평균 제출의 Public Macro F1은 **0.4489813603**이다.
이전 최고였던 exp012의 `0.4388787816`보다 **+0.0101025787**, exp011 최초
enrichment 챔피언 `0.4352596431`보다 **+0.0137217172** 상승했다. 3-seed OOF
평균 `0.5398447261`과 Public LB의 절대 gap은 `-0.0908633658`이다.

따라서 exp14는 현재 **로컬 OOF 및 Public LB 챔피언**이다. 단일 LGBM 점수는
LR보다 낮았지만, LR과 다른 오류를 내는 LGBM 및 train-discovered specialist를
20% 혼합한 다양성 전략이 실제 test에서도 유효했다.

현재 20% weight는 seed 42 OOF grid에서 선택한 탐색 결과다. 팀의 엄격한 최종
검증 계약을 적용할 때는 outer-fold train 내부 OOF에서 weight를 고르고 outer
validation에 적용하는 nested 검증이 추가로 필요하다.

## 제출 파일 생성

현재 확인된 exp14 후보를 seeds 42/52/62로 full-train 재학습하고 확률 평균한다.

`experiment.ipynb`의 **10. 제출 파일 생성** 셀 두 개를 위에서부터 실행한다.

출력:

```text
experiments/SDH/exp_014_focal_lgbm_specialists/results/
  submission_exp014_safe_lr80_lgbm20_3seed.csv
```

각 seed에서 다음 순서로 학습한다.

1. 전체 train으로 안전 exp13 vocabulary, recurrent mutation, enrichment를 fit
2. test에는 고정된 변환을 적용만 수행
3. 전체 train mutation prevalence로 유사 암종쌍 2개 자동 발견
4. LR과 balanced multiclass LGBM 및 두 binary specialist 학습
5. hard routing 후 `LR 80% + LGBM 20%`
6. 세 seed 확률 평균 후 최종 클래스 선택

노트북 셀은 `C__`, `D__exact` 피처 부재, train/test 비결합 audit, ID 순서,
중복 ID, 예측 결측, 확률합을 검사한다. 현재 20% weight는 OOF 탐색값이므로
nested weight 검증 전 제출은 비교용 후보로 취급한다. 실제 Public LB는
**0.4489813603**으로 확인됐다.
