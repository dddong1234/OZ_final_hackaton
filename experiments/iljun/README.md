# member_d (권일준) 작업 폴더

담당 트랙: *(미정 — 스탠드업에서 확정)*

## 지금 여기 있는 것

| 경로 | 내용 |
|---|---|
| `EDA_REPORT.md` | **EDA 상세 보고서** — 팀원용. 노션 【📊 EDA 상세 보고서】와 같은 내용 |
| `notebooks/01_eda.ipynb` | EDA + 베이스라인 (정본). Run All 하면 2분 |
| `notebooks/artifacts/` | 표·그림·spec |
| `results/member-d-logreg-001/` | 기준 실험 — Macro F1 **0.36305** |

---

## 네이밍 규칙

파일이 계속 늘어나므로 아래 규칙을 지킵니다. **규칙이 길면 안 지켜지므로 최소한만 둡니다.**

### 1. 실험 ID — 모든 기록의 공통 키

```
member-d-<모델>-<번호3자리>
```

- 예: `member-d-logreg-001` · `member-d-lgbm-003` · `member-d-blend-001`
- 번호는 **내 것만** 001부터. 팀 공통 일련번호가 아닙니다
- 모델 토큰은 아래 고정 목록에서만 고릅니다

| 토큰 | 모델 |
|---|---|
| `logreg` | LogisticRegression |
| `lgbm` `xgb` `cat` | LightGBM · XGBoost · CatBoost |
| `rf` `mlp` `svm` `knn` | RandomForest · MLP · SVM · kNN |
| `blend` | 가중 평균 앙상블 |
| `stack` | 스태킹 |

> **트랙 정보는 ID에 넣지 않습니다.** 노션 【📋 실험·제출 기록】의 `트랙` 컬럼에 넣습니다.
> ID가 길어지면 결국 아무도 안 지킵니다.

### 2. 노트북 — 앞자리가 트랙

```
notebooks/
├─ 00_quick_start.ipynb      팀장 제공 · 수정하지 않음
├─ 01_eda.ipynb              공통 기반 (EDA · 베이스라인 · 전처리 spec)
├─ 1x_a_*.ipynb              트랙 A · 변이 표기 활용
├─ 2x_b_*.ipynb              트랙 B · 저신호 클래스 구제
├─ 3x_c_*.ipynb              트랙 C · 중첩 코호트 판별
├─ 4x_d_*.ipynb              트랙 D · 피처 선택 · 검증 · 앙상블
├─ 9x_scratch_*.ipynb        임시 · 커밋하지 않음
└─ _archive/                 폐기. 지우지는 않되 정본과 헷갈리지 않게
```

예: `11_a_hotspot_encoding.ipynb` · `21_b_combo_marker.ipynb` · `41_d_l1_selection.ipynb`

**첫 markdown 셀에 실험 메타를 적습니다** (팀장 README 규약).

```markdown
# 핫스팟 인코딩

- Owner: member_d
- Experiment ID: member-d-lgbm-003
- Track: A
- Seed: 42
- Validation: StratifiedKFold-5
- 목적: BRAF V600E/V600K 를 개별 피처로 뺐을 때 THCA F1 이 오르는가
```

### 3. artifacts — 노트북 번호를 접두로

```
notebooks/artifacts/<노트북번호>_<섹션>_<이름>.<확장자>
```

- 예: `01_s2_class_counts.csv` · `01_s4_tmb.png` · `11_a_hotspot_lift.csv`
- 노트북 안에서 `NB = "01"` 을 정의하고 `ARTIFACTS / f"{NB}_..."` 로 저장합니다
- 노트북이 늘어나도 파일이 섞이지 않고, 어느 노트북이 만든 건지 이름만 봐도 압니다

### 4. features — 트랙 산출물

```
features/features_<트랙>.py        예: features_A.py
features/artifacts/<트랙>_spec.json
```

`build_features(df) -> (X, feature_names)` 인터페이스를 지킵니다.
`feature_names` 에는 `A_` 처럼 트랙 접두를 붙여 충돌을 막습니다.

### 5. results — 실험 ID 그대로

```
results/<실험ID>/
├─ metrics.json        공유 대상 (커밋)
├─ submission.csv      커밋하지 않음
└─ model.joblib        커밋하지 않음
```

---

## git 에 올리는 것 / 올리지 않는 것

| 대상 | 커밋 |
|---|---|
| `EDA_REPORT.md` · `README.md` | ✅ |
| `notebooks/*.ipynb` (00·01·1x\~4x) | ✅ |
| `notebooks/9x_scratch_*.ipynb` | ❌ 임시 |
| `notebooks/artifacts/*.csv` `*.png` | ✅ 표·그림은 팀원이 봐야 함 |
| `notebooks/artifacts/01_preprocess_spec_v1.json` | ❌ 41KB · 노트북 실행하면 재생성 |
| `notebooks/artifacts/01_environment.json` | ✅ 작아서 |
| `features/*.py` | ✅ |
| `results/*/metrics.json` | ✅ |
| `results/*/submission.csv` `model.joblib` | ❌ `.gitignore` 가 이미 제외 |

> 루트 `.gitignore` 에는 `artifacts/` 규칙이 없습니다. 공용 파일이라 임의로 고치지 않고
> **`git add` 할 때 위 표대로 골라 담습니다.** 규칙으로 굳히려면 팀장에게 공지 후 한 줄 추가.

### 브랜치 · PR (팀장 `docs/GIT_STRATEGY.md` 3.1 / 12.2 준수)

내 팀원 식별자는 **`iljun`** 이다. 브랜치·PR 제목·커밋 어디서나 이 하나만 쓴다.

```text
iljun/exp-<세 자리 번호>-<실험내용>

예) iljun/exp-001-eda
    iljun/exp-002-hotspot-encoding
    iljun/exp-003-class-weight-tuning
```

- 번호는 **세 자리로 통일** (`exp-1` `exp-01` 은 규약 위반)
- 실험내용은 영문 소문자 + 하이픈. 한글·공백 금지
- **새 실험 브랜치는 항상 최신 `main` 에서** 생성한다 (체인식 분기 금지)
- `main` 직푸시 금지. 브랜치 push 후 PR

PR 제목:

```text
[iljun][EXP-001] Add EDA report and reproducible baseline notebook
```

PR 본문은 `docs/GIT_STRATEGY.md` 12.2 템플릿을 따른다 —
실험 목적 / 주요 변경사항 / Validation / 재현 방법 / 커밋 제외 파일 / 참고사항.

---

## 점수 기록 규칙 (협업 규정 2)

- **소수점 5자리로 기록**, 비교할 때는 **넷째 자리에서 반올림해 셋째 자리까지만**
- `0.36276` 과 `0.36305` 는 둘 다 `0.363` → **동점**
- **Macro F1 과 Accuracy 를 항상 함께** 기록 (`class_weight` 제거 시 Macro F1 −0.032 / Accuracy −0.009)
- `seed` 와 `validation` 을 반드시 기입 — 노션 DB에 전용 컬럼이 있습니다
