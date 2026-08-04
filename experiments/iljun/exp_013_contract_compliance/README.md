# exp_013 — 팀 계약 준수화: 고정 도메인 피처 제거 + 가중치 fold-local 재선택

실험자 권일준 · 2026-08-04 · 브랜치 `iljun/exp-013-contract-compliance`

점수를 올리는 실험이 **아니다.** 팀 Notion `팀 모델 분업·안전한 앙상블 운영 계약`과
2026-08-04 exp13 안전 baseline 공지에 우리 앙상블 라인을 맞추고, 그 과정에서
**규정 준수 비용이 얼마인지**를 실측했다.

결론부터 — **비용이 음수다.** 규정 위험 요소를 전부 제거하니 앙상블 점수가 오르고
seed 간 분산이 줄었다.

---

## 1. 무엇이 문제였나

계약의 누수 방지 표와 공지가 우리 구성의 세 지점을 짚는다.

| # | 조항 | 우리 상태 |
|---|---|---|
| 1 | `train+test concat 으로 vocabulary 를 만들지 않는다` | 앙상블 제출 경로 2곳이 위반 |
| 2 | `논문·DB 의 암종쌍을 모델 입력·규칙으로 쓰지 않는다` | `KIRC/KIPAN`, `LGG/GBMLGG` 고정 |
| 3 | `고정 exact mutation 금지` (2026-08-04 공지) | `BRAF V600E` 등 4개 고정 |
| 4 | `가중치는 outer fold 의 train split 내부 OOF 에서만 고른다` | 전체 OOF 스윕으로 선택 |

1번은 `exp_011_model_ensemble/` 두 파일을 `make_submission_context` 로 교체해 해소했고
예측 불변(0/2,546행)까지 실측했다. 2~4번이 이 폴더의 내용이다.

---

## 2. 고정 contrast → fold-train 자동 발견 (`run_contrast_comparison.py`)

fold 마다 `discover_confusion_pairs`(fold-train 3-fold 대리모델 혼동행렬 상위 8쌍)로
쌍을 다시 발견한다. 구현은 감사받은 `final_pipeline/final_submission.py` 것을 불러 쓰고,
SDH·gs 폴더는 건드리지 않는다(협업 규정 4) — 우리 프로세스 안에서 candidate 를
`dataclasses.replace` 로 갈아끼운다.

seeds 42/52/62, 같은 split.

| 구성 | champion 평균 | blend 평균 | blend σ | 피처 평균 |
|---|---:|---:|---:|---:|
| 고정 contrast + 고정 exact (legacy) | 0.527429 | 0.538883 | 0.002250 | ~8,200 |
| 자동 contrast + 고정 exact | 0.526989 | 0.541019 | 0.002113 | ~8,210 |
| **자동 contrast + exact 제거 (준수)** | 0.526859 | **0.541354** | **0.001725** | 8,206.2 |

- **자동 발견 교체: blend `+0.002135`, 3/3 seed 양수.** champion 단독은 `-0.000440` 로 동률.
- **고정 exact 4개 제거: champion `-0.000130`, blend `+0.000335`.** 비용이 없다.
- legacy → 완전 준수 합계 **blend `+0.002471`**, 분산은 0.002250 → 0.001725 로 감소.

### exp_013 의 `-0.002393` 과 다른 이유

exp13 README 는 고정 contrast **제거** 비용을 쟀다. 여기서는 **교체**를 쟀다.
제거 대신 fold-train 자동 발견으로 바꾸면 그 비용이 사라지고 앙상블에서는 이득이다.

### 팀장 exp14 와의 대조

발견 방법이 다른데(우리 = fold-train 혼동행렬, 팀장 = prevalence 코사인) 결론이 같다.

| | 방법 | 앙상블 이득 | seed |
|---|---|---:|---|
| 우리 | 혼동행렬 상위 8쌍 | +0.002135 | 3/3 |
| 팀장 exp14 | prevalence 코사인 상위 2쌍 | +0.001793 | 3/3 |

---

## 3. 가중치 fold-local 재선택 (`run_foldlocal_blend.py`)

계약대로 outer fold 마다 그 fold 의 train split 안에서 inner 3-fold OOF 를 만들고
**거기서만** 가중치를 고른 뒤 해당 outer validation 에 적용한다. 전체 outer OOF 는
점수를 적는 용도로만 쓴다.

계약 grid 는 2-model 전용이라 두 가지를 모두 보고한다 — 엄격 준수(LR+파트너 1개,
계약 5점)와 확장(three-way 사전 선언 9점, 실행 전 코드에 고정).

| 구성 | Macro F1 평균 | σ | anchor 대비 | 양수 |
|---|---:|---:|---:|:---:|
| anchor (LR 단독) | 0.526989 | 0.005729 | — | — |
| fold-local LR+ovr | 0.529235 | 0.005513 | +0.002247 | 3/3 |
| fold-local LR+lgbm | 0.536755 | 0.004221 | +0.009766 | 3/3 |
| **fold-local three_way** | **0.537729** | **0.002100** | **+0.010740** | **3/3** |
| 계약 이전 고정 0.55/.30/.15 | 0.541019 | 0.002113 | +0.014030 | 3/3 |

### 선택 편향 실측 `+0.003290`

전체 OOF 에서 고른 가중치가 fold-local 대비 그만큼 부풀려 나온다. 이 값이 permutation
감사의 잡음 천장 `+0.006845`(231점 격자, `exp_012_cross_member_ensemble/artifacts/`)
안에 들어오므로 두 측정이 서로 모순되지 않는다.

**보고 숫자를 바꿔야 한다.** 기존 ENS-011a 이득 `+0.01608`(clean seed) 대신
**`+0.0107`** 를 쓴다. 편향을 걷어내도 3/3 seed·σ 0.0021 로 이득 자체는 남는다.

### 부수 관찰

- **LGBM 이 이득의 91%** — `LR+lgbm` 단독 +0.009766 vs three_way +0.010740.
  새 분업에서 LGBM 은 신동훈님 담당이므로, 우리는 anchor LR 을 지키는 쪽이 맞다.
- **three_way 의 가치는 점수가 아니라 안정성** — σ 가 0.004221 → 0.002100 으로 절반.
  OVR 은 점수를 거의 안 올리지만 seed 간 흔들림을 줄인다.

---

## 4. 확률 파일 계약 포맷 (`export_probabilities.py`)

역할 분담상 공통 split/OOF 확률 관리가 우리 몫인데, 우리 OOF 는 ID·fold 열이 없는
`.npy` 배열로만 있었다. `MODEL_DIVERSITY_STRATEGY.md` §2.3 포맷으로 맞췄다.

```text
OOF  : ID, SUBCLASS, seed, fold, prob_<class×26>
test : ID, seed, prob_<class×26>
```

OOF 는 2절이 저장한 npz 를 재사용한다 — `StratifiedKFold(5, shuffle, random_state=seed)`
가 결정적이라 fold 번호를 정확히 복원할 수 있어 **모델을 다시 학습하지 않는다.**
test 확률만 새로 만들며 어휘는 `make_submission_context`(train 전용)를 쓴다.

3 모델 × 3 seed × (OOF + test) = 18 파일 + `manifest.json`.
§7.2 확률 정규성(유한값, 행 합 1±1e-6)을 쓰기 전에 assert 로 건다.
확률 파일은 §2.3 대로 `results/` 아래(gitignore)에만 두고 커밋하지 않는다.

---

## 5. 우리 숫자를 팀 baseline 과 직접 비교하지 말 것

우리는 SDH exp_012 모듈 + gs B04 경로이고 팀장은 exp13 standalone 독립 구현이다.

| | 팀장 | 우리 |
|---|---:|---:|
| 안전 LR baseline | 0.527609 | 0.526859 |
| 안전 앙상블 | 0.539845 (LR 80 + LGBM 20) | 0.541354 (three_way, legacy 가중치) |
| CV 피처 수 | 8,193.2 | 8,206.2 |

같은 대역이지만 구현이 다르다. 우리 0.541354 는 legacy 가중치 기준이므로 fold-local
보정(-0.003290)을 적용하면 약 **0.5381** 로 읽어야 한다.

---

## 6. 파일

| 파일 | 내용 |
|---|---|
| `run_contrast_comparison.py` | 고정 vs 자동 contrast, `--drop-exact` 로 고정 exact 제거 |
| `run_foldlocal_blend.py` | 계약대로 fold-local 가중치 선택 + 선택 편향 실측 |
| `export_probabilities.py` | OOF/test 확률을 계약 포맷으로 내보내기 |
| `artifacts/contrast_comparison.json` | 고정 vs 자동 (exact 유지) |
| `artifacts/contrast_comparison_noexact.json` | 자동 + exact 제거 (규정 준수 구성) |
| `artifacts/foldlocal_blend_auto.json` | fold-local 가중치, fold 별 선택 기록 포함 |

---

## 7. 남은 것

- **LB 슬롯** — 규정 준수 구성은 예측이 바뀌므로 확인에 슬롯이 하나 필요하다.
- **inner enrichment seed 규약 불일치** — exp14 는 `outer_seed × 100 + fold`,
  우리는 `inner_seed = seed` 고정이다. 누수 문제는 아니지만 재현 규약이 갈리므로
  팀 표준을 정하는 게 좋다. 우리가 맞추면 한 줄 변경이다.
- **cross-member 후보 3개** — `exp_012` 에서 잰 경수님 라인 이득(+0.011~+0.012)은
  전체 OOF 에서 고른 값이라 이 폴더의 편향 실측(+0.003290)만큼 깎아서 읽어야 한다.
  채택하려면 fold-local 절차로 재측정해야 한다.
