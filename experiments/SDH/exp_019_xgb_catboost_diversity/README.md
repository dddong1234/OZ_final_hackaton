# SDH exp_019 — XGBoost / CatBoost model diversity

## 한 줄 요약

B10의 안전 표현을 고정하고 XGBoost 12개, CatBoost 12개를 모델별 입력 view까지
달리해 스크리닝한다. 목표는 LR을 단독으로 이기는 모델 하나가 아니라 **기존
LR+LGBM 앙상블 위에서 순수한 추가 이득을 내는 후보**를 찾는 것이다.

초기 설계의 `LR 80% + 후보 20%` 판정은 폐기했다. 단독 성능·상관·LR과의 blend가
좋아도 기존 LGBM과 오류 역할이 중복되면 최종 앙상블을 하락시킬 수 있기 때문이다.

## 기준 계약

- 원본: train만 읽어 outer Stratified 5-fold 검증
- seeds: 42 스크리닝 후 42/52/62 확인
- 표현: B04 `G+B+V+T+R+A+S` + gene-type enrichment
- contrast: outer-fold train의 inner 3-fold 대리 LR 혼동행렬에서 상위 8쌍 자동 발견
- exact: 고정 exact mutation 없음
- pair/group: 고정 암종쌍·고정 유전자군 없음
- vocabulary/support/표준화/hybrid 선택: outer-fold train에서만 fit
- 음수: enrichment와 signed auto contrast를 그대로 유지
- test: 이 실험에서는 읽지 않음

exp13 standalone parser를 안전한 행별 파싱 기반으로 사용하되, auto contrast와 모델별
feature view는 exp19에서 직접 구현한다. raw train/validation을 결합하지 않는다.

## 왜 ComplementNB처럼 음수를 제거하지 않나

현재 행렬의 음수는 결측이나 오류가 아니다.

- enrichment 음수: 해당 변이가 특정 암종에 상대적으로 적다는 정보
- contrast 음수: 자동 발견된 두 암종 중 반대 방향 유전자 신호

XGBoost와 CatBoost는 signed numeric feature를 처리할 수 있으므로 clipping·shift를
하지 않는다. train mean보다 낮다는 방향 정보도 그대로 모델에 전달한다.

## 네 가지 입력 view

### full

약 8.2천 개 전체 안전 sparse 피처. LR과 같은 정보를 비선형 트리로 학습한다.

### compact

다음 저차원 구조·집계 블록만 유지한다.

```text
B + V + A-pair + S + auto contrast + enrichment
+ truncating/recurrent 전체 count
```

수천 개 희소 유전자 열을 제거해 트리가 안정적인 집계값에 집중하는지 본다.

### hybrid512 / hybrid1024

compact에 outer-fold train support가 높은 `G/T/R` 열 512개 또는 1,024개를 추가한다.
어떤 열을 고를지는 validation을 보지 않고 fold-train nonzero count로만 결정한다.

## 24개 case

### XGBoost 12개

| case | view | 주요 차이 |
| --- | --- | --- |
| x01~x04 | full | depth 3/5/7, 느린 700-tree 대조 |
| x05~x06 | compact | 저차원 집계 전용 |
| x07~x08 | hybrid512 | depth 3/5 |
| x09 | hybrid1024 | 중간 복잡도 |
| x10 | full | 강한 L1/L2 |
| x11 | full | subsample/column sample 강화 |
| x12 | hybrid1024 | DART dropout booster |

### CatBoost 12개

| case | view | 주요 차이 |
| --- | --- | --- |
| c01~c04 | full | symmetric depth 4/6/8, 800-tree 대조 |
| c05~c07 | compact | depth 4/6/8 |
| c08~c09 | hybrid512 | depth 4/6 |
| c10 | hybrid1024 | 중간 복잡도 |
| c11 | full | 강한 L2·낮은 random strength |
| c12 | hybrid1024 | 높은 random strength·bagging temperature |

두 모델 모두 fold-train 클래스 역빈도 sample weight를 사용한다. XGBoost의
멀티클래스에 부적절한 `scale_pos_weight`는 사용하지 않는다.

## 실행 순서

이미 XGBoost/CatBoost 스크리닝을 시작했다면 커널을 재시작하지 않는다. 기존
`screen_results`를 그대로 두고 노트북 **마지막 수정 셀**을 실행한다.

1. 데이터·패키지·24개 case 확인
2. seed42 안전 피처와 네 view 준비
3. seed42 LR anchor 학습
4. XGBoost 12개 스크리닝
5. CatBoost 12개 스크리닝
6. 기존 B10 LGBM outer OOF 5회 추가 학습
7. 고정 `LR80+LGBM20` 위에 후보 10%를 넣는 값으로 싸게 재정렬
8. family별 1위만 outer-train 내부 3-fold에서 LR/LGBM/후보 weight 선택
9. strict seed42 증분이 양수인 후보만 3-seed 확인 대상으로 승격

6~9는 `incremental_recheck.py`가 담당한다. 고정 10% 결과는 후보 순위용일 뿐
채택 근거가 아니며, 최종 판정에는 strict outer fold-local 결과만 사용한다.

## 비교 지표

- 단독 Macro F1
- 기존 LGBM과 예측 불일치율 및 확률 상관
- 기존 `LR+LGBM` 오답 복구 수
- 기존 `LR+LGBM` 정답 훼손 수
- 순복구 수 = 복구 수 - 훼손 수
- 둘 중 하나가 맞으면 정답을 고르는 oracle Macro F1
- strict fold-local `LR+LGBM` 기준 Macro F1
- strict fold-local `LR+LGBM+후보` Macro F1과 증분
- 후보가 0%로 선택된 fold 수와 fold별 증분

상관이 낮다는 사실만으로 후보를 채택하지 않는다. 새로 맞히는 행보다 새로 틀리는
행이 많으면 기각한다. 전체 OOF에서 가장 좋은 weight를 사후 선택한 값도 판정에
사용하지 않는다.

## 승격 기준

seed42에서 다음 조건을 모두 만족해야 3-seed strict 확인으로 승격한다.

- `LR+LGBM+후보` strict incremental delta > 0
- 5개 outer fold 중 최소 3개에서 delta > 0
- 모든 fold에서 후보 weight가 0%로 선택되지는 않음
- 순복구 수와 선택 weight를 함께 확인
- 수치·확률·행 순서·fold audit 통과

단독 점수가 높거나 상관이 낮아도 strict incremental delta가 음수면 기각한다.

## 예상 실행량

- seed42 준비: 5 outer folds, 각 fold auto-pair inner 3-fold + enrichment inner 5-fold
- seed42 screen: 24 × 5 = 120 model fits
- cheap 재판정: 기존 LGBM 5 fits만 추가
- strict seed42: 공유 inner 안전 표현/LR/LGBM 15 fits + family 1위마다 15 fits
- 3-seed 확장은 strict seed42 통과 후보가 있을 때만 별도 진행

CatBoost full depth8과 XGBoost DART가 가장 오래 걸린다. 노트북은 XGBoost와
CatBoost 스크리닝 셀을 분리했으므로 중간에 멈췄다가 이어갈 수 있다.

## 확정 결과 — seed 42

24개 case와 수정된 incremental 재판정이 오류 없이 완료됐다.

### 단독 모델

| 구분 | 1위 case | view | OOF Macro F1 |
| --- | --- | --- | ---: |
| XGBoost | `x01_full_d3_400` | full | **0.502834** |
| CatBoost | `c03_full_d8_500` | full | **0.490572** |

단독 1위가 최종 추가 후보 1위는 아니었다. 기존 LGBM 위에서 역할이 겹치는지를
반영하면 XGBoost는 `x12_hybrid1024_dart`, CatBoost는 `c01_full_d4_500`이
family별 strict 확인 대상으로 선택됐다.

### 저비용 고정 weight 재정렬

기준은 고정 `LR80+LGBM20 = 0.542541`이다. 이 기준선의 90%와 후보 확률 10%를
결합한 값은 순위 결정에만 사용했다.

| 후보 | 단독 F1 | 고정 결합 F1 | 기준 대비 | LGBM 상관 | LGBM 불일치 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `x12_hybrid1024_dart` | 0.490436 | **0.543674** | **+0.001133** | 0.903 | 30.3% |
| `x11_full_d5_random` | 0.497472 | 0.543142 | +0.000601 | 0.922 | 26.8% |
| `c01_full_d4_500` | 0.473806 | 0.542499 | -0.000041 | 0.796 | 43.1% |
| `c09_hybrid512_d6` | 0.489661 | 0.542437 | -0.000104 | 0.839 | 38.5% |

CatBoost는 LGBM과 더 다르지만 고정 결합에서는 family 전체가 기준선을 넘지
못했다. 상관이나 불일치율을 단독 채택 기준으로 사용할 수 없다는 사례다.

### Strict outer fold-local incremental 결과

각 outer fold의 train 내부에서만 inner 3-fold로 weight를 고른 후, 해당 outer
validation에 적용했다.

| 후보 | LR+LGBM 기준 | 후보 추가 | 증분 | 양수 fold | 후보 평균 weight | 복구/훼손 | 순복구 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `x12_hybrid1024_dart` | 0.541299 | **0.541665** | **+0.000366** | 3/5 | 0.08 | 43/43 | 0 |
| `c01_full_d4_500` | 0.541299 | **0.541489** | **+0.000190** | 3/5 | 0.13 | 44/60 | -16 |

XGBoost DART의 fold별 증분:

```text
-0.002932, +0.000270, 0.000000, +0.000606, +0.004612
```

CatBoost의 fold별 증분:

```text
-0.005175, -0.000827, +0.001147, +0.004604, +0.002943
```

두 후보 모두 사전 승격 조건인 양수 증분과 3/5 양수 fold를 충족했다. 다만 개선폭이
`+0.0004`보다 작고 앞쪽 fold에서 함께 하락했다. XGBoost는 전체 정답 수가 그대로인
상태에서 클래스별 배분이 좋아졌고, CatBoost는 정답 수를 16개 잃으면서 Macro F1만
소폭 상승했다.

## 판정

- **즉시 채택: 없음**
- **3-seed 확인 1순위:** `x12_hybrid1024_dart`
- **보류:** `c01_full_d4_500`; XGBoost 확인 후 계산 여유가 있을 때만 수행
- XGBoost DART가 3-seed에서 일관되게 양수가 아니면 GBDT 추가 축은 최종 기각

고정 기준선 `0.542541`은 공유받은 GBDT 스크리닝 기준선 `0.542466`과
`+0.000075` 차이로 거의 일치했다. 반면 exp19의 strict 기준선은 `0.541299`로
약 `-0.00117` 낮다. inner split seed 또는 fold-local 전처리 구현 차이일 수 있으므로
팀 간 최종 수치 비교 때는 같은 OOF artifact와 fold 계약을 사용해야 한다.

## 한계

- strict 결과는 seed 42 단일 검증이다.
- 24개 후보 중 family별 한 개만 strict 재검증했다.
- 작은 증분은 seed·fold 변화로 쉽게 뒤집힐 수 있다.
- 전체 OOF best weight나 Public LB는 후보 선택에 사용하지 않았다.
- `screen_errors.json`과 `seed42_strict_foldlocal_errors.json`은 모두 비어 있다.
