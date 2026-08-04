# exp_009 — 제출 코드 규칙 감사 + train/test 어휘 커버리지

실험자 권일준 · 2026-08-03 · 브랜치 `iljun/exp-009-leakage-audit-and-coverage`

점수를 올리는 실험이 **아니다.** 팀 1위 제출본(`exp-gs-002-08`, LB 0.38711)이 코드 검증에서
걸릴 수 있는 지점 세 곳을 제거하고, 그 과정에서 드러난 표현 수준별 전이율을 정량화했다.

> **갱신 2026-08-03** — 이후 SDH exp_011(class-enrichment)이 LB 0.43525 로 챔피언이 됐다.
> exp_011 은 GS 원본 candidate 를 불러 쓰므로 아래 세 가지를 **그대로 상속**했다.
> 이 폴더의 `run_submission.py` 는 셋 다 고치고 예측 불변까지 실측한 판본이므로,
> 최종 제출은 그 위에 enrichment 를 얹는 것이 맞다. 5.3 절에 exp_011 누수 감사 결과가 있다.

---

## 1. 왜 했나

시상 기준은 리더보드 90% 인데, 그 앞에 **수상 제외 조건**이 있다 — Data Leakage 적발,
코드 재현 불가, 제출 규정 미준수. 1위와 격차가 0.00223 인 상황에서는 점수 +0.002 를 버는
것보다 실격 리스크를 0 으로 만드는 쪽이 기대값이 크다.

`exp-gs-002-final_single_run.py` 를 읽고 세 곳을 찾았다.

| # | 위치 | 문제 | 조치 |
|---|---|---|---|
| 1 | `main()` | `assert test NaN == 237` — test 실측값을 코드에 고정 | assert 제거, 참고용 print 로 강등 |
| 2 | `make_submission()` | `pd.concat([train, test])` — 팀 규정 5 자가점검이 명시 금지한 패턴 | train/test 분리 파싱으로 재구성 |
| 3 | 메타데이터 | `"leakage_check": True` 가 계산 없는 상수 | 실측 검사로 교체 |

**1번이 가장 위험하다.** 평가 환경의 test 파일이 조금이라도 다르면 `AssertionError` 로
실행이 멈춘다 — "코드 동작 여부"와 "Score 재현 가능 여부"는 평가 항목이고, 재현 불가는
수상 제외 사유다. 동시에 test 를 사전 관찰했다는 근거로도 읽힌다.

**2번은 실질 누수가 아니었다.** 학습되는 선택이 전부 `train_index` 로 마스킹돼 있고
(`active` / `trunc` / `recurrent` / contrast / `nonconstant_columns`), test 에만 있는
이벤트 열은 train 에서 전부 0 이라 recurrent(>=5) 필터와 상수열 제거에서 자동 탈락한다.
다만 **검증자가 그걸 확인해 줄 의무는 없다.** 제출본에서 가장 먼저 grep 될 문자열이라
구조를 바꿔 없앴다.

---

## 2. 무엇을 바꿨나

`run_submission.py` — gs 원본의 사본에 위 세 가지를 적용했다. gs 폴더는 건드리지 않았다
(협업 규정 4).

핵심 변경은 어휘 분리다.

```python
# 이전: train+test 를 붙여 한 번에 파싱 → 어휘가 train ∪ test
combined = pd.concat([train[genes], test[genes]], axis=0, ignore_index=True)
cache = RowCache.build(combined, genes)

# 이후: train 이 어휘를 정의하고, test 는 그 어휘에 투영
train_cache = RowCache.build(train[genes], genes)
test_cache  = RowCache.build(test[genes], genes, vocabulary=train_cache.event_names)
cache       = RowCache.stack(train_cache, test_cache)
```

`burden` / `variant` / `amino` / `topology` 는 행 내부 집계라 test 자기 이벤트를 전부
세는 동작이 그대로 유지된다. 어휘에 투영되는 것은 `event_matrix` 뿐이다.

**실측 누수 검사** — test 행이 아예 없는 캐시로 train 설계행렬을 다시 만들어 완전 일치를
assert 한다. 어떤 test 행이라도 어휘·선택·스케일에 영향을 줬다면 두 행렬이 달라진다.

---

## 3. 동치성 검증

세 수정이 예측을 바꾸지 않았음을 실측했다.

```
✅ 완전 일치 — 2546행 전부 동일한 예측
   피처 수 8,399 (원본과 동일) · 수렴 경고 0건
```

`artifacts/equivalence.json`. 재현:

```bash
.venv/bin/python experiments/gs/notebooks/submission/exp-gs-002-final_single_run.py \
    --submission-name baseline_original.csv
.venv/bin/python experiments/iljun/exp_009_leakage_audit_and_coverage/run_submission.py \
    --submission-name patched.csv
.venv/bin/python experiments/iljun/exp_009_leakage_audit_and_coverage/compare_submissions.py \
    experiments/iljun/results/baseline_original.csv \
    experiments/iljun/results/patched.csv
```

---

## 4. 부수 발견 — 표현 수준별 전이율

어휘를 분리하면서 train/test 겹침을 직접 셀 수 있게 됐다. `run_vocab_coverage.py` 로 측정.

| 표현 수준 | test 커버리지 | 비고 |
|---|---|---|
| 정확 변이 (종류 기준) | **5.1%** | 13,910 / 273,514 |
| 정확 변이 (출현 기준) | **5.5%** | 18,714 / 337,294 |
| recurrent missense 블록 | 0.42% | 230 열이 test 환자 43.5% 를 건드림 |
| 유전자 단위 (G 블록) | **96.9%** | 4,225 / 4,358 |

**test 변이의 94.5% 는 train 에 한 번도 나오지 않는다.** 출현 횟수 기준으로도 같으므로
희귀 싱글턴 문제가 아니라 표현 수준의 문제다. 유전자 단위로 내려가면 96.9% 가 전이된다.

### 팀 관측과의 대조

이 수치가 그동안의 관측을 설명한다.

- **CV<->LB 간격(0.08~0.14)** — CV 는 train 내부라 어휘가 겹치지만 LB 는 94.5% 가 미지의
  변이다. 튜닝 부족이 아니라 구조적 간격이다.
- **정확토큰(D) 패스스루가 65% 에 그친 것** — 토큰이 test 에서 거의 켜지지 않는다.
- **A_pair 가 단독 최대 리프트(+0.037)였던 것** — 380 차원으로 압축한 치환 *방향* 은
  어휘 의존이 없어 처음 보는 변이에도 켜진다.
- **`LR-002k`(집계만, G 제외)가 0.157 에 그친 것** — G 가 96.9% 커버리지를 지탱하는
  유일한 축이었다.

### 함의

남은 헤드룸은 정확 변이를 더 넣는 쪽이 아니라 **저차원 전이 가능 공간으로 압축하는 쪽**에
있다. exp_008 의 자동 혼동쌍(쌍당 2 열)과 홍주님 S 블록(표기 구조 9 열, CV<->LB 간격
0.082 로 최소)이 둘 다 이 성질을 갖는다.

---

## 5. 이 커버리지 틀로 판정한 것들

커버리지 측정이 예측 도구로 쓰인다는 것이 확인됐다. 제출 슬롯·seed 를 쓰기 전에
후보를 거를 수 있다.

### 5.1 CV 재현 (`run_cv.py`)

동치성 검증은 '전체 train 학습 → test 예측' 경로만 봤으므로 CV 도 따로 쟀다.

| 항목 | 이번 실행 | gs 기준값 | 차이 |
|---|---|---|---|
| OOF Macro F1 (3seed) | 0.47850165 | 0.47850165 | **+0.0000000000** |
| 피처 수 평균 | 8173.53 | 8173.53 | — |

gs 의 CV 경로(`exp-gs-002-memory-safe.py:753`)는 원래부터 train-only 였다. concat 은
제출 생성 경로에만 있었으므로 CV 가 안 바뀌는 것이 정상이고, 이 실행이 확인했다.

주의 — gs 기준값의 표준편차는 **ddof=1**(표본표준편차)이다. `np.std` 기본값(ddof=0)
으로 재면 같은 데이터에서 0.002028 vs 0.002484 로 갈린다. 팀 내 통일이 필요하다.

### 5.2 경수님 Event Ontology (`run_ontology_coverage.py`, `run_signature.py`)

세 후보의 운용 지점 test 커버리지를 먼저 재서 하나만 3-seed 로 돌렸다.

| 블록 | CV 증분(seed42) | test 출현 커버 | 판정 |
|---|---|---|---|
| gene × position | +0.003030 | **~1.5%** | 사전 기각 (CV-only 아티팩트 프로필) |
| gene × 50aa-bin | +0.001317 | ~70% | 사전 기각 (CV 이득 미검출) |
| gene × functype × bin | **+0.004032** | ~35% | 3-seed 확인 → **⚠️ 미검출** |

signature 3-seed: seed42 **+0.004723** (경수님 +0.004032 재현), seed2024 +0.004718,
seed777 **−0.002453** → 평균 +0.00233 ± 0.00414, 2/3 양수. SE 0.00239 로 1σ 미만.
팀 선례와 일관 — `LR-002j`(2/3·σ안) 기각, `LR-008a`(3/3·σ초과) 채택.

### 5.3 SDH exp_011 supervised FE 감사 (`run_permutation_check.py`)

class-enrichment(LB 0.43525)는 label 로 피처를 만드는 supervised FE 라 누수 위험이
가장 큰 범주다. label 을 섞고 같은 파이프라인을 돌렸다.

| 조건 | OOF Macro F1 | enrichment 이득 |
|---|---|---|
| 실제 label + base | 0.478142 | — |
| 실제 label + enrichment | **0.529491** | **+0.051349** |
| 섞인 label + base | 0.031996 | — |
| 섞인 label + enrichment | 0.033034 | **+0.001038** |

**50배 차 → ✅ PASS.** cross-fit 이 깨졌다면 섞인 label 도 되찾아 큰 이득이 났어야
한다. 재현도 됐다(제 0.529491 vs SDH 보고 0.52640) — 문서의 *안전한 절차만으로*
이득이 재현된다는 것 자체가 이득의 출처가 누수가 아니라는 증거다.

⚠ 검증한 것은 SDH 실제 코드가 아니라 `TEAM_REPORT.md` 에 적힌 절차다. 구현이 문서와
다른 지점이 있으면 못 잡는다. SDH 님이 본인 코드로 재실행해야 완성된다.

### 5.4 커버리지가 팀 실패 기록을 설명한다

| 실험 | 축 | 커버 | 결과 |
|---|---|---|---|
| exp_005 코돈 | 위치 | 낮음 | 기각 |
| exp_008 위치구간 | 위치 | 낮음 | 기각 |
| exp_009 signature | 위치(bin) 포함 | ~35% | 미검출 |
| exp_011 Case05 exact-event enrichment | 정확 변이 | **0.3%** | 기각 |
| **exp_011 Case04 gene×type enrichment** | **위치 없음** | **81%** | **+0.049 채택** |

어휘를 폭발시키는 축(위치·정확 AA)을 빼는 것이 전이의 조건이었다.

---

## 6. 파일

| 파일 | 내용 |
|---|---|
| `run_submission.py` | 규칙 감사를 반영한 단일 파일 제출 러너 (gs 원본의 사본 + 3 수정) |
| `run_cv.py` | 3-seed CV 재현 — gs 기준값 대조 |
| `run_vocab_coverage.py` | 표현 수준별 커버리지 측정 |
| `run_ontology_coverage.py` | 후보 블록별 test 전이율 (제출 전 후보 거르기용) |
| `run_signature.py` | 경수님 signature 블록 3-seed paired 검증 |
| `run_permutation_check.py` | supervised FE 누수 감사 (label 셔플) |
| `compare_submissions.py` | 두 제출 CSV 완전 일치 대조 |
| `artifacts/coverage.json` | 표현 수준별 커버리지 |
| `artifacts/equivalence.json` | 동치성 검증 결과 |
| `artifacts/cv.json` | 3-seed CV 재현 |
| `artifacts/ontology_coverage.json` | 온톨로지 후보별 커버리지 |
| `artifacts/signature.json` | signature 3-seed paired delta |
| `artifacts/permutation_check.json` | permutation 감사 결과 |

## 7. 남은 것

- **환경** — 검증은 Python 3.14 에서 했다. 레포 규약은 3.12 이므로 최종 제출 코드에 기재할
  환경은 3.12 로 다시 맞춰야 한다. (동치성 검증은 같은 환경 내 전후 비교라 영향 없음)
- **하드코딩된 코호트 쌍** — `FINAL_CONTRAST_PAIRS = (("KIRC","KIPAN",5), ("LGG","GBMLGG",5))`
  는 이번 변경에서 손대지 않았다. exp_008 의 자동 발견판으로 교체하면 규칙 3 리스크가
  사라지지만 **예측이 바뀌므로** 별도 LB 슬롯이 필요하다.
- **공용 승격** — 이 러너를 `final_pipeline/` 로 올릴지는 팀 결정 사항이다. 공용 변경은
  사전 공지 + `shared/...` 브랜치가 규정이다.
