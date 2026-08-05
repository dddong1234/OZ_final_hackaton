# 26개 암종 분류 실험 정리: exp_model_004 ~ 현재

## 한 줄 결론

현재 규정 안전 실험의 핵심 성과는 **gene×functional event type을 암종별 Empirical-Bayes evidence 26점수로 압축한 P1+EB**이다. 이후의 큰 구조 전환은 대부분 이 기준선을 넘지 못했으며, 현재는 **faithful H0를 주력으로 두고 XGBoost가 LR/LGBM specialist가 놓치는 비선형 오류를 보완하는지**를 검증하는 단계다.

> 주의: 팀의 Public LB `0.45348` 3-way ensemble은 train/test concat을 포함해 데이터 누수 가능성이 확인되었다. 따라서 아래의 규정 안전 성능 비교 및 제출 후보에서는 제외하고 참고 수치로만 보관한다.

---

## 1. 공통 규칙과 평가 기준

| 항목 | 고정 원칙 |
|---|---|
| 데이터 사용 | OOF 실험에서는 train만 사용하고 test는 읽지 않음 |
| 검증 | Stratified 5-fold, 기본 seeds `42 / 777 / 2024` |
| 지표 | OOF Macro F1 |
| supervised FE | event vocabulary, recurrent event, EB 통계, 자동 specialist 후보는 outer-fold train에서만 fit |
| 변환 | validation/test에는 학습된 변환만 적용 |
| 결측 | WT·빈 문자열·NaN은 mutation event가 아님 (`nan_as_mutation_count=0`) |
| 금지 | train-test concat, test 기반 encoding/scaling/통계, 고정 암종·유전자·exact mutation 규칙 |
| 모델 기준 | LR은 주력 판별기, 트리/확률 모델은 오류 다양성이 검증될 때만 보완 모델로 사용 |

**해석 원칙:** 로컬 OOF만으로 제출 모델을 확정하지 않는다. 안전한 OOF에서 후보를 미리 고정한 뒤, 제한된 제출로 LB 일반화 격차를 확인한다. LB에 맞춘 반복적인 weight·threshold 재탐색은 하지 않는다.

---

## 2. 출발점: P1+Empirical-Bayes

P1은 샘플의 개별 mutation binary를 그대로 넓게 사용하는 대신, `gene × functional event type`이 각 암종을 얼마나 지지하는지 fold-train에서 계산하고 **암종별 26개 dense evidence score**로 압축한다.

Empirical-Bayes(EB)는 희귀 token을 단순 제거하지 않고, 전체 발생률 쪽으로 보수적으로 수축해 희귀 token의 과대해석을 줄인다.

| 기준 | 3-seed OOF Macro F1 | 결과 |
|---|---:|---|
| 구조화 LR | `0.526222 ± 0.001234` | 비교 기준 |
| P1 + EB LR | **`0.533739 ± 0.001667`** | **채택**; 평균 `+0.007517`, 세 seed 모두 상승 |

P1+EB는 실제 제출에서 Public LB `0.4299354648`을 기록했다. 이는 현재 **규정 안전하게 LB가 확인된 본인 제출 기준선**이다.

---

## 3. exp_model_004 — P1+EB 취약 구간과 잔차 보정

### 3-1. P1+EB 취약 구간 감사

목적은 새 피처를 바로 더하는 것이 아니라, EB가 어떤 샘플에서 개선/악화되는지 train-only OOF로 찾는 것이었다.

| 관찰 | 결과 | 의미 |
|---|---|---|
| 낮은 margin `<0.05` | 약 1,550행, 세 seed에서 EB가 구조화 LR보다 하락 | EB의 evidence가 애매한 샘플에서는 잘못된 암종을 밀 수 있음 |
| 변이 수 `11+` | 약 3,713행에서 EB가 일관되게 상승 | 전체 개선은 변이가 풍부한 샘플에서 주로 발생 |
| 변이 수 `1`, `6~10` | 대체로 하락/비개선 | 저변이 샘플은 정보 부족 또는 EB 과신 구간 |
| profile 중복/신규 | 신규 profile에서도 개선 | 단순 중복 profile이 일반화 격차의 주원인이라는 근거는 약함 |

감사 재현 OOF는 structured LR `0.527550/0.526006/0.525111`, P1+EB `0.533907/0.534100/0.531846`이었다. 모든 seed에서 누수 검사 통과, NaN mutation 0, 수렴 경고 0을 확인했다.

### 3-2. Selective EB gate

확인된 취약 구간만 보수적으로 복귀시키는 고정 규칙을 검증했다.

- P1+EB margin `<0.05`이면 P1 non-EB 확률 사용
- 그 외에는 P1+EB 확률 사용
- threshold `0.05`는 discovery seeds에서 한 번 정하고, 새 seeds `31415 / 52 / 62`에서 재검증

| 후보 | 새 3-seed OOF | P1+EB 대비 | 상태 |
|---|---:|---:|---|
| P1+EB | `0.527958 ± 0.001625` | — | 기준 |
| selective EB gate | **`0.534446 ± 0.001027`** | **`+0.006488 ± 0.002610`** | 검증 후보 |

세 새 seed 모두 상승했다. 다만 이 보고서 시점에는 최종 15-fold 상승 수·클래스별 붕괴 확인 및 LB 제출 비교가 완료 기록으로 확정되지 않았으므로, **P1+EB를 대체한 최종 제출 모델로는 아직 확정하지 않는다.**

### 3-3. All-class evidence ranker

환자마다 26개 암종 후보의 찬성/반대 EB evidence 통계 45개를 비교해 전체 후보 순위를 다시 매기는 모델을 seed42에서 검증했다.

| 후보 | seed42 OOF | gate 대비 | 해석 |
|---|---:|---:|---|
| selective EB gate | `0.536009` | — | 기준 |
| all-class ranker | **`0.551792`** | **`+0.015783`** | 단일 seed 신호 |

Top-3 recall도 `0.762619 → 0.784712`로 높아졌다. 반면 low-margin Macro F1은 `0.238036 → 0.224926`으로 하락했고 3-seed 검증 전이다. 따라서 점프 가능성은 있었으나, **단일 seed 결과만으로 제출 후보로 승격하지 않았다.**

### 3-4. EB-offset sparse residual model

EB score를 유지한 채 raw sparse token으로 잔차만 보정하려 했으나 성능이 하락했다.

| 후보 | seed42 OOF | gate 대비 | 판정 |
|---|---:|---:|---|
| EB-offset residual | `0.523379` | `-0.012182` | 기각 |

**인사이트:** EB 위에 고차원 raw residual을 다시 얹는 방식은 안정적인 EB 신호를 흐릴 가능성이 높다.

### 3-5. Parser grammar recovery

mutation 원문 표기와 복수 이벤트 표기를 복원해 `UNKNOWN`을 줄이는 방향을 감사했다.

- 개선 parser audit: raw segment `247,778`, parsed event `247,778`, unknown `0`, multi-event cell `19,016`
- segment conservation은 통과했으며 NaN/WT는 event가 되지 않았다.

그러나 parser 교체 단독 비교는 기존 P1+EB `0.534391` 대비 `0.532173` (`-0.002218`)이었다.

**판정:** 문법 복원 자체는 데이터 품질·감사 측면에서 유효하지만, 현 분류 성능을 올리는 축으로는 미검출이다. event-type 세분화와 위치/복합 구조 확장은 우선순위를 낮췄다.

---

## 4. exp_model_005 — 모델 패러다임 전환

### 4-1. Raw vs normalized profile purity 감사

원본 표기 형식에 classifier가 버린 정보가 있는지 확인했다.

| profile 표현 | unique profiles | weighted purity | conflict profiles |
|---|---:|---:|---:|
| raw | 5,636 | 0.917594 | 447 |
| normalized | 5,636 | 0.917594 | 447 |

**인사이트:** 현재 normalisation이 표기 방식 정보를 실질적으로 지우지 않았다. raw delimiter/prefix/order를 별도 모델 축으로 확장할 근거가 없다.

### 4-2. Frozen biomedical encoder

제공된 gene·event type·위치·ref/alt 문자열만 고정 사전학습 encoder에 넣고, 환자별 pooling embedding을 LR로 분류했다. 외부 annotation·sequence는 사용하지 않았다.

| 후보 | seed42 OOF | P1+EB 대비 | 판정 |
|---|---:|---:|---|
| frozen encoder 단독 | `0.311034` | `-0.222752` | 기각 |
| fixed blend | `0.508218` | `-0.025567` | 기각 |

**인사이트:** 짧은 mutation 표기에 대한 frozen 자연어 embedding은 현재의 암종별 evidence 표현을 보완하지 못했다.

### 4-3. Class-conditional Evidence Set Network

P1+EB가 합산하며 잃는 evidence의 분포 형태(강한 한 개 vs 약한 여러 개, 찬성/반대 evidence)를 후보 암종별 집합 모델로 학습했다.

| 후보 | seed42 OOF | 비교 기준 | 판정 |
|---|---:|---:|---|
| evidence set network | `0.297658` | team reference `0.542020` 대비 `-0.244362` | 기각 |

구조는 규정 안전했고 train-only nested EB를 사용했으나, 6,201행으로 event-set scorer를 안정적으로 학습하기에는 신호가 부족했다.

---

## 5. exp_model_006 — faithful H0 복원 및 LR 보완 후보

### 5-1. Faithful H0 reproduction

기존 실험의 기준선 차이를 제거하기 위해 exp013/014를 GS 내부 self-contained 코드로 재현했다.

H0 구성:

- 구조화 mutation FE: mutation binary, burden, event-type count, truncation, fold-train recurrent missense, A-pair log1p, S topology, gene×event-type enrichment
- multiclass LR
- multiclass LGBM + fold-train 자동 유사 암종쌍 hard specialist
- 최종 확률: `0.80 LR + 0.20 specialist LGBM`

| 구성 | seed42 OOF | reference 대비 | 판정 |
|---|---:|---:|---|
| exp013 LR | `0.525910` | `-0.000220` | 재현 |
| exp014 LGBM hard specialist | `0.493092` | `+0.000760` | 재현 |
| **faithful H0 blend** | **`0.544744`** | `+0.001065` | 안전 기준선 |

H0 3-seed 평균은 `0.542587 ± 0.003365`이다. 이는 P1+EB보다 OOF가 높지만, 안전한 H0 제출의 LB는 이 보고서 시점에 확인되지 않았다.

### 5-2. Auto confusion MoE

26개 클래스를 inner OOF 혼동으로 6개 그룹으로 묶고 그룹 specialist로 확률 질량을 재분배했다.

| 후보 | seed42 OOF | H0 대비 | 판정 |
|---|---:|---:|---|
| auto confusion MoE | `0.509085` | `-0.035659` | 기각 |

그룹이 과도하게 커져(주요 그룹 약 21 classes) specialist가 오히려 전체 경계를 약화시켰다.

### 5-3. Evidence-shape pairwise ranker (H2-S)

EB evidence의 양/음/집중도 등 19개 feature로 모든 암종 후보를 pairwise 재순위화했다.

`H0 0.523717`, `H2-S 0.525561`로 내부 delta는 `+0.001844`였으나 H0가 faithful 기준 `0.543679`보다 `-0.019962` 낮아 **baseline reproduction failure**로 판정했다.

**결론:** 후보 성능과 무관하게 기준선 재현 실패 시 비교를 무효 처리하는 원칙을 확정했다.

### 5-4. Profile retrieval

fold-train 내 동일 mutation profile의 label 분포를 lookup으로 활용했으나 validation match rate가 약 11~13%에 불과했다.

| 후보 | seed42 OOF | H0 대비 | 판정 |
|---|---:|---:|---|
| profile retrieval blend | `0.537723` | `-0.007021` | 기각 |

### 5-5. Safe 3-way ensemble 재현

multinomial LR, OVR LR, base LGBM을 안전하게 재현해 기존 3-way ensemble과 비교했다.

| 후보 | seed42 OOF | H0 대비 | 판정 |
|---|---:|---:|---|
| safe 3-way | `0.537492` | `-0.007252` | 기각 |

누수 가능성이 있던 LB `0.45348` 앙상블은 이 안전 재현에서는 재현되지 않았다. 따라서 이전 LB 개선을 정상 모델 성능 근거로 사용하지 않는다.

### 5-6. Auto-validated pair specialist

inner OOF에서 암종쌍 specialist의 개선을 자동 검증한 뒤, 통과한 쌍만 outer validation에 적용했다.

| 후보 | seed42 OOF | H0 대비 | 판정 |
|---|---:|---:|---|
| auto-validated pair specialist | `0.546385` | `+0.001641` | 미검출 |

low-margin F1은 `+0.005686` 개선됐지만, 전체 delta가 작고 3-seed 확정 조건에 미달해 채택하지 않았다.

### 5-7. Intragenic architecture EB

같은 유전자 안의 복수 이벤트 수·same-codon·position span 등을 EB score로 압축했다.

| 후보 | 3-seed OOF | H0 대비 | 판정 |
|---|---:|---:|---|
| intragenic architecture EB | `0.540836 ± 0.003008` | `-0.001750 ± 0.000412` | 기각 |

---

## 6. exp_model_007 — 현재 진행 중인 보완 모델

### 6-1. Complement NB profile blend

faithful H0와 독립적인 `gene×event-type` Complement NB profile을 outer-fold train에서만 만들고, 고정 `0.75 H0 + 0.25 NB`를 3-seed 검증하도록 준비했다.

- 목적: 선형/트리 H0와 다른 class-conditional profile 오류를 보완
- 규칙: vocabulary는 fold-train에서만 생성, validation-only token은 무시, test 미열람
- 상태: **실행 대기 또는 실행 중** — 결과가 아직 이 보고서에 없음

### 6-2. H0 + XGBoost complement

현재 준비한 실험이다.

- 입력: H0가 fold-train 안에서 생성한 규정 안전 구조화·EB sparse matrix
- XGB: `multi:softprob`, 300 trees, learning rate `0.03`, depth `4`, 강한 L1/L2 정규화, histogram tree
- 결합: `0.80 H0 + 0.20 XGB` 고정
- 단계: seed42 screen 후 통과 시 `42/777/2024` 3-seed 확정
- screen 승격: H0 대비 `+0.008` 이상, 5 folds 중 4개 이상 상승
- 상태: **실행 대기 또는 실행 중**

XGB는 LR을 대체하려는 목적이 아니다. LR/H0가 놓치는 비선형 경계를 보완하는지, 그리고 그 보완이 3-seed에서 안정적인지를 확인하는 실험이다.

---

## 7. 현재 채택 상태와 제출 판단

| 구분 | 후보 | 근거 | 상태 |
|---|---|---|---|
| 실제 안전 LB 기준 | P1+EB | Public LB `0.4299354648` | 현재 본인 제출 기준 |
| OOF 안전 기준 | faithful H0 | 3-seed `0.542587 ± 0.003365` | LB 비교 필요 |
| 조건부 후보 | selective EB gate | 새 3-seed `+0.006488` | 최종 fold/class 점검 및 LB 비교 필요 |
| 실험 중 | H0+Complement NB, H0+XGB | 오류 다양성 검증 목적 | 결과 대기 |
| 제외 | 누수 3-way LB `0.45348` | train/test concat | 참고만, 제출 금지 |

---

## 8. 팀 공유용 인사이트 및 다음 행동

1. **가장 강한 안전 FE는 암종별 EB evidence score다.** 개별 mutation 열을 무작정 늘리는 것보다, gene×event-type의 암종별 지지도를 26개 score로 압축하는 방식이 큰 개선을 만들었다.
2. **원본 mutation을 바로 학습하는 복잡한 모델은 아직 실패했다.** frozen encoder, evidence set network, sparse residual은 EB/H0보다 크게 낮았다. 새 모델은 EB/H0를 버리지 말고 보완해야 한다.
3. **baseline faithful reproduction이 선행 조건이다.** H2-S처럼 기준선이 0.02 낮으면 candidate의 개선 수치는 해석할 수 없다.
4. **OOF와 LB를 분리해 관리한다.** safe OOF로 사전 후보를 고정하고, 정해진 후보만 제한적으로 LB에서 비교한다. 누수 가능 앙상블의 LB는 목표치로 사용하지 않는다.
5. **다음 우선순위는 오류 다양성이다.** H0+Complement NB와 H0+XGB가 H0보다 단독으로 낮더라도, fixed blend가 모든 seed에서 안정적으로 상승하면 제출 후보가 된다.

---

## 재현/근거 파일

- `exp_model_004/result/`: vulnerability audit, selective gate, parser, ranker, residual 결과
- `exp_model_005/result/`: raw profile audit, frozen encoder, evidence set 결과
- `exp_model_006/result/`: faithful H0, specialist, architecture, safe ensemble 결과
- `exp_model_007/common/`: Complement NB 및 XGBoost 보완 실험 실행기

모든 표의 수치는 각 실험의 저장된 summary CSV 및 leakage audit JSON에서 인용했다.
