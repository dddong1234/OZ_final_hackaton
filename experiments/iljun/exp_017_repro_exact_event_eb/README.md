# exp_017 — Exact-event EB 3-seed 재현 검증

LB **0.5086030051 (1위)** 제출물 `submission_h0_exact_event_eb_seed42_777_2024_bagged.csv` 를
**제3자가 독립적으로 재현할 수 있는지** 확인하기 위한 폴더다.

새로운 모델을 만들지 않는다. 채점 대상 코드를 그대로 돌려 **같은 CSV가 나오는지**만 본다.

---

## 1. 파일 구성

| 파일 | 설명 |
| --- | --- |
| `reproduce_exact_event_eb_3seed.py` | **원본: 경수(gs)님 작성.** 무수정 사본. 규정 대응 v2 (2026-08-06) |
| `compare_submissions.py` | 내 재현 결과와 원본 제출 CSV의 일치율·audit 상수 대조 |
| `run_repro.sh` | smoke → seed42 → 3seed → compare 순차 실행 러너 |
| `out/` | 재현 산출물. `.gitignore` 처리 (GIT_STRATEGY 2-6) |

> `reproduce_exact_event_eb_3seed.py` 는 gs 폴더의 정본이 push 되면 그쪽을 정본으로 삼고
> 이 사본은 삭제한다. 현재 main 에는 exact-event 버전이 아직 없어 재현용으로만 동봉했다.

---

## 2. ⚠️ 인자 없이 그냥 돌리면 안 되는 이유

원본 스크립트는 `--output-dir` 를 안 주면

- 출력 폴더 기본값 = `experiments/gs/notebooks/submission/`
- 출력 파일명 기본값 = `submission_h0_exact_event_eb_seed42_777_2024_bagged.csv`

즉 **LB 0.5086 을 올린 그 파일을 덮어쓴다.**
`run_repro.sh` 는 항상 `--output-dir` 를 이 폴더의 `out/` 으로 강제하므로, 러너를 통해 실행할 것.

---

## 3. 실행 절차

```bash
cd <repo-root>
pip install lightgbm scikit-learn scipy pandas numpy   # .venv 사용 권장

# 1) smoke — 수 초. test.csv 를 읽지 않는다. 경로/파서 확인용
bash experiments/iljun/exp_017_repro_exact_event_eb/run_repro.sh smoke

# 2) 단일 seed — 시간 측정. 여기서 나온 시간 × 3 이 3-seed 예상 소요
bash experiments/iljun/exp_017_repro_exact_event_eb/run_repro.sh seed42

# 3) 3-seed 전체 재현
bash experiments/iljun/exp_017_repro_exact_event_eb/run_repro.sh 3seed

# 4) 원본 제출물과 비교
bash experiments/iljun/exp_017_repro_exact_event_eb/run_repro.sh compare
```

`.venv` 가 레포 루트에 없으면 `PY=/path/to/python bash run_repro.sh ...` 로 지정한다.

---

## 4. 재현 성공 판정 기준 (사전 등록)

결과를 보고 기준을 만들지 않기 위해 **미리 못박는다.**

| 일치율 | 판정 | 해석 |
| ---: | --- | --- |
| 100% | 완전 재현 | 코드와 환경 모두 동일 |
| ≥ 99% | 사실상 재현 | 26-class argmax 는 확률 1e-6 차이로도 몇 행이 뒤집힌다. 라이브러리 버전 차이로 본다 |
| 95–99% | 조사 필요 | audit 상수를 대조해 어느 단계에서 갈렸는지 특정 |
| < 95% | 재현 실패 | 실행 환경 또는 입력 데이터가 다르다 |

`compare_submissions.py` 는 불일치 시 다음 audit 필드를 자동 대조한다.

- `exact_vocabulary_size` 가 다르면 → 파서/입력 데이터 단계
- `final_feature_count`, `structured_feature_count` 가 다르면 → 설계행렬 단계
- `specialist_pairs` 가 다르면 → centroid cosine 자동 탐색 단계
- 위가 전부 같은데 예측만 다르면 → 순수 라이브러리 버전 차이

---

## 5. 코드 검토 결과 요약 (v1 → v2)

모델 로직 구간(`normalise_cell` ~ `exact_eb_features`)은 **문자열 단위로 v1 과 완전히 동일**함을 확인했다.
변경은 전부 모델 바깥이며, 따라서 **LB 0.5086 은 그대로 유효하다.**

| 변경 | 내용 |
| --- | --- |
| 삭제 | 미사용 `REFERENCE_LR / REFERENCE_BLEND / REFERENCE_TOLERANCE` |
| 추가 | `environment_metadata()` — OS·라이브러리 버전, `pretrained_model_used: False` |
| 추가 | `data_directory()` — `--data-dir` → `/data` 자동감지 → `data/raw` |
| 추가 | `submission_directory()` — `--output-dir`, 레포 밖 복사 시 fallback |
| 추가 | `# -*- coding: utf-8 -*-`, `--data-dir` / `--output-dir` CLI 인자 |

### 규정 대조 (전 항목 통과)

| 규정 | 근거 |
| --- | --- |
| seed·하이퍼파라미터가 코드 상수로 공개 | `RECURRENT_MIN_COUNT`, `EB_*`, LR/LGBM 파라미터, seed 하드 assert |
| 수동 암종/유전자/변이 목록 없음 | 유일한 리터럴은 파서 단위테스트의 `"R132H R132H"` (중복 제거 검증용) |
| exact-event vocabulary 는 train 자동 생성 | `fit_vocabulary(train_frame)` 한 곳에서만 생성 |
| test 는 변환·추론만 | `keep` 마스크를 `x_train` 기준으로만 계산, train/test concat 없음 |
| 외부데이터·사전학습·모델파일 없음 | `pretrained_model_used: False`, `external_model_file_required: False` |
| 한 `.py` 로 제출까지 재현 | 레포 import 0개 |

### 남은 개선 여지 (재현과는 무관, 제출 안정성 관점)

1. `make_submission_frame` 이 `sample_submission.ID` 순서가 test 와 다르면 raise 한다.
   3-seed 학습을 다 마친 뒤 마지막 줄에서 터질 수 있으므로, raise 대신 ID 기준 매핑이 안전하다.
2. `print()` 안의 `×` 문자(L623)는 비-UTF8 로케일 stdout 에서 `UnicodeEncodeError` 가능성이 있다.
3. `evaluate_h0`, `run` 은 정의만 되고 호출되지 않는다(AST 확인). 규정 위반은 아니나,
   `evaluate_h0` 는 docstring 에 "OOF 재현용, 제출 경로에서는 미사용" 을 명시해 두면 오해가 없다.

---

## 6. 기록할 것

3-seed 실행 후 다음을 Notion TEST log 에 남긴다.

- 일치율과 판정
- seed42 단일 소요 시간, 3-seed 총 소요 시간
- `environment` audit 필드(파이썬·numpy·pandas·scipy·sklearn·lightgbm 버전)
