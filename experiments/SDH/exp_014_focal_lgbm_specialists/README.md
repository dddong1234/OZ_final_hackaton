# SDH exp_014 — focal LGBM and pair specialists

## 실험 질문

exp13 standalone 챔피언 피처를 고정한 상태에서 focal-loss LGBM과 주요 혼동
암종쌍 전용 LGBM이 단일 Macro F1 또는 LR 앙상블 다양성을 개선하는가?

## 공통 조건

- 전처리: exp13 standalone 8,425개 피처
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

## 2단계 — pair specialist

seed 42 메인 승자를 고른 후 binary LGBM specialist를 fold-train의 해당 클래스
행만으로 학습한다.

- KIRC/KIPAN: 10 trees, lr 0.1, leaves 20, min child 20
- LGG/GBMLGG: 100 trees, lr 0.02, leaves 20, min child 10
- 쌍별 단독, 두 쌍 soft mass α=0.15/0.30/0.50
- predicted-only soft 0.30 및 hard routing

fixed pair는 기존 train OOF 오류 분석 기반 연구 가설이다. 유전자 선택과
specialist 학습에는 validation/test label을 사용하지 않는다.

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
| multiclass balanced | **0.481370** |
| focal γ=1 | 0.472474 |
| focal γ=1 + balanced | 0.472474 |
| multiclass unweighted | 0.471061 |
| focal γ=2 | 0.465645 |

Focal loss는 채택하지 않고 일반 multiclass balanced를 메인으로 선택했다.

### Specialist seed 42

두 암종쌍 predicted-only hard routing이 메인 LGBM을
`0.481370 → 0.494918`로 `+0.013548` 개선해 specialist 승자로 선택됐다.

### LR 80% + LGBM 계열 20%

| Seed | exp13 LR | LGBM + hard specialist | Blend | LR 대비 |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0.529185 | 0.494918 | **0.540164** | +0.010979 |
| 52 | 0.529905 | 0.493890 | **0.536910** | +0.007005 |
| 62 | 0.525617 | 0.504001 | **0.537084** | +0.011467 |
| 평균 | 0.528236 | 0.497603 | **0.538052** | **+0.009817** |

3개 seed 모두에서 개선됐다. LGBM 단독 성능은 LR보다 낮지만 seed 42에서 LR
오답 634행을 복구했고 확률 상관은 0.7674로 충분한 다양성을 보였다. Public LB는
확인하지 않고, exp15에서 LGBM 전용 피처 공간을 먼저 탐색한다.
