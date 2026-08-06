# Experiment Log — 26개 암종 분류

기준 지표는 OOF Macro F1이며, 별도 표기가 없는 점수는 Stratified 5-fold 결과다. `3-seed`는 seeds `42 / 777 / 2024`의 평균±표준편차다. 모든 채택 후보는 fold-train only 통계·피처 선택, `leakage_check=True`, `nan_as_mutation_count=0`을 공통 계약으로 사용했다.

| 실험 | 모델 | Feature Engineering | 주요 파라미터 | Score | 비고 |
|---|---|---|---|---:|---|
| Baseline | Multinomial Logistic Regression | WT=0, mutation=1 binary gene | `C=0.07`, `max_iter=2000`, balanced, 5-fold | 0.344689 (seed42) | 초기 기준선 |
| DOMAIN-04A | LR / LightGBM | 암종군 count, exact mutation, gene-pair, 선택 gene mutation | LR 기본 / LGBM 500 trees | LR 0.340109±0.008119, LGBM 0.297241±0.002243 | binary baseline 대비 소폭 개선 |
| Hongju H-A | Multinomial LR | G+B+V+T+R+A: gene binary, burden, event type, truncation, recurrent missense, A-pair | `C=0.07`, `max_iter=2000`, balanced | 0.430525±0.004421 | A-pair 효과 확인 |
| Hongju H-AS | Multinomial LR | H-A + S topology/diversity/entropy | 동일 | 0.431366±0.003318 | 이후 초반 backbone |
| H-AS + exact hotspot 4개 | Multinomial LR | H-AS + BRAF/IDH1/PIK3CA exact feature 4개 | 동일 | 0.433479±0.002436 | 채택; 고정 hotspot은 이후 팀 규칙상 사용 중단 |
| H-AS + gene-pair | Multinomial LR | H-AS + IDH1–TP53, IDH1–ATRX, APC–TP53 | 동일 | 0.431332±0.001664 | 미검출 |
| H-AS exact ablation | Multinomial LR | exact hotspot 4개 개별 제거 | 동일 | 기록 미확인 | exact block 기여 확인용 |
| B count binning | Multinomial LR | 행별 mutatio
n count를 고정 12-bin one-hot | 동일 | 0.480266±0.000783 | 평균 +0.001765이나 seed777 하락 → 미검출 |
| KIRC↔KIPAN soft specialist | LR + binary specialist | primary 확률의 pair 내부 비율만 soft 보정 | `α=0.30` | 0.478576±0.002564 | +0.000075 → 미검출 |
| Event TF-IDF | TF-IDF LR + primary | G/E/TYPE/AA mutation 문서 TF-IDF, 0.5 확률 평균 | fixed 0.5 blend | 0.480305 | TF-IDF 단독 0.375735, 앙상블 채택 |
| Event TF-IDF OVR | OVR TF-IDF LR + primary | 같은 mutation 문서, One-vs-Rest 분류 | 0.5 blend | 0.483130±0.002638 | best; 이후 LB는 하락 |
| TF-IDF 3-way | LR + multinomial/OVR TF-IDF | primary 0.50 + multinomial 0.25 + OVR 0.25 | 고정 blend | 0.481853±0.002430 | 기각 |
| Sparse FM screen | PyTorch Factorization Machine + LR | 08 구조화 sparse feature | rank=8, LR=3e-4, balanced, LR:FM=0.75:0.25 | seed42 0.486993 | screen 후보 |
| Sparse FM 3-seed | FM + LR | 위 FM 설정 고정 | rank=8, LR=3e-4 | 0.483024±0.003517 | 평균 +0.001010, seed777 하락 → 미검출 |
| Complement NB profile blend | Complement NB + LR | mutation-profile 기반 class probability | LR:NB=0.75:0.25 | 0.489167±0.001189 | 3 seed 모두 상승, 채택 |
| P1 gene×event-type enrichment | Multinomial LR | fold-train gene×functional-type log-odds를 26차원 class score로 압축 | inner cross-fit, `C=0.07` | **0.526222±0.001234** | +0.044208, 큰 전환 축 |
| Nested stacking | Meta LR | P1 + NB-ratio OVR + Complement NB raw probability 78개 | nested cross-fit | 0.529305±0.004500 | 일부 seed 하락 → 미검출 |
| gene×A-pair enrichment | Multinomial LR | gene×ref→alt class score | seed42 screen | 0.476761 | 기각 |
| 위치 계층 enrichment | Multinomial LR | gene×type×50aa position-bin EB | seed42 screen | 0.527929 | +0.000379 → 미검출 |
| Dense enrichment LGBM | LightGBM | 26 enrichment score + 저차원 burden/type/topology | shallow regularized LGBM | 0.381396 | 기각 |
| Co-mutation pair enrichment | Multinomial LR | recurrent gene-pair class EB score 26개 | fold-train support≥10 | 0.528354 | 미검출 |
| P1 OVR | One-vs-Rest LR | P1 입력 동일, 목적함수만 OVR | `C=0.07` | 0.521450 | 기각 |
| Empirical-Bayes enrichment | Multinomial LR | 희귀 gene×event-type도 전역 prior 쪽으로 posterior shrinkage | α=1, shrinkage=20, clip=4 | **0.533739±0.001667** | P1 대비 +0.007517, 3 seed 모두 상승 |
| Point-process EB | Multinomial LR | 연속 위치·allele evidence 78개 | seed42 screen | 0.518255 | 기각 |
| Multivariate EB | Multinomial LR | class effect 저랭크 분해 | rank=4 / 8 | 0.483194 / 0.496296 | 기각 |
| Frozen biomedical encoder | Frozen transformer + LR | 제공 mutation 문자열 mean/max embedding | fixed encoder | 0.311034 | 기각; pretrained 모델 축 종료 |
| Parser grammar recovery | Multinomial LR | multi-event 문법 복원 및 canonical type 확장 | parser 단독 비교 | 0.532173 (seed42) | 기준 0.534391 대비 하락 → 종료 |
| Selective EB gate | LR branch gate | EB margin<0.05에서는 non-EB LR, 나머지는 EB LR | threshold=0.05, 새 seeds 31415/52/62 | **0.534446±0.001027** | EB 대비 +0.006488, 채택 |
| All-class evidence ranker | Candidate LR ranker | 후보 암종별 EB evidence shape 45개 | inner OOF candidate ranking | 0.551792 (seed42) | H0 재현 불일치로 미승격 |
| EB-offset residual | Sparse linear residual LR | EB score offset + raw token correction | strong regularization | 0.523379 | 기각 |
| Evidence Set Network | shared event-set network | 후보 암종별 event evidence pooling | PyTorch, inner OOF EB | 0.297658 | 기각 |
| H0 automatic specialist | LR + LGBM specialist | 자동 발견 상위 유사 class-pair 2개, pair mass 보존 | LR 0.80 + specialist LGBM 0.20 | 0.547915 (seed42) | 규정 안전 H0 |
| H1 auto confusion MoE | LR + group LGBM MoE | inner OOF 혼동 기반 6개 자동 group | LGBM 100 trees, LR=0.02 | 0.512330 | group이 과도하게 커져 기각 |
| H0 component complement | H0 + non-EB/EB branch | H0와 alternate branch 동일 가중 확률 혼합 | equal blend | 0.548144 (seed42) | +0.000229, 작은 안정화 신호 |
| Fold-aligned bagging audit | H0 3-seed bagging | 동일 outer split 안에서 model seeds 42/777/2024 평균 | equal weights | 0.546091 | seed42 0.547915 대비 -0.001823; bagging은 안정성 감사 목적 |
| Exact-event EB | H0 Selective-EB LR + LGBM specialist | **train 자동 gene×exact-event vocabulary**의 26차원 posterior EB score 추가 | 3 seeds, α=1, shrinkage=20, clip=4 | **0.568441±0.002310** | H0 대비 +0.021186, 15/15 fold 상승, 현재 로컬 best |
| Exact-event EB confidence | Multinomial LR | exact EB + evidence confidence/shape 234개 | seed42 screen | 0.558086 | baseline 대비 -0.012067; DLBC 큰 하락 → 기각 |
| Exact-event EB 3-seed submission | 3-seed bagged H0 + Exact-event EB | full-train seed 42/777/2024 equal probability average | fixed validated configuration | **Public LB 0.5086** | 현재 최고 제출 기록 |

## 공통 재현 조건

| 항목 | 내용 |
|---|---|
| 데이터 | Train 약 6,201행, Test 약 2,546행, 유전자 4,384개, 26-class |
| 목표 지표 | Macro F1 |
| 기본 LR | `LogisticRegression(solver='lbfgs', C=0.07, max_iter=2000, class_weight='balanced')` |
| 기본 검증 | Stratified 5-fold, seeds 42/777/2024 |
| 누수 방지 | vocabulary·support·EB 통계·표준화·specialist 선택을 fold-train에서만 fit |
| 결측 처리 | WT/빈 문자열/NaN은 event 0개; test NaN을 mutation으로 만들지 않음 |
| 현재 제출 파이프라인 | H0 구조화 feature + gene×event-type EB + 자동 exact-event EB + selective EB gate + 자동 LGBM specialist + 3-seed bagging |

## 현재 상태

- **현재 로컬 best**: Exact-event EB, 3-seed OOF `0.568441 ± 0.002310`
- **현재 최고 Public LB**: Exact-event EB 3-seed bagging `0.5086`
- **다음 원칙**: 자동 선택·fold-train only를 유지하고, 고정 암종/유전자/변이 목록을 코드에 사용하지 않는다.
