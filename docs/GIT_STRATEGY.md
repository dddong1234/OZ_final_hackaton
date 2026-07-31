# Git 브랜치 및 실험 협업 전략

## 1. 목적

이 프로젝트는 여러 팀원이 각자 독립적으로 AI 실험을 진행하고, 재현 가능하고 공유할 가치가 있는 결과만 `main` 브랜치에 반영하는 방식으로 운영한다.

브랜치 간 의존성과 불필요한 충돌을 줄이기 위해 다음 원칙을 적용한다.

> 모든 개인 실험은 원칙적으로 최신 `main`에서 독립적인 브랜치를 생성하여 진행한다.

---

## 2. 핵심 원칙

1. `main`에서 직접 실험하거나 커밋하지 않는다.
2. 실험 하나당 브랜치 하나를 생성한다.
3. 새로운 실험 브랜치는 최신 `main`에서 생성한다.
4. 이전 실험 브랜치에서 다음 실험 브랜치를 만드는 체인식 분기는 지양한다.
5. 개인 실험과 공용 구조 변경은 별도 브랜치로 분리한다.
6. 원본 데이터, 모델, checkpoint, submission 및 예측 확률 파일은 커밋하지 않는다.
7. 좋은 결과는 재현 가능한 코드와 설정으로 정리한 뒤 Pull Request를 생성한다.
8. 실패하거나 채택하지 않은 실험은 원칙적으로 코드 대신 실험 지식을 Notion에 공유한다.
9. PR이 `main`에 병합되면 팀 채팅에 공유한다.
10. 팀원은 새로운 작업을 시작하기 전에 로컬 `main`을 최신화한다.

---

## 3. 브랜치 네이밍 규칙

브랜치 이름은 영문 소문자, 숫자, 하이픈을 사용한다. 공백과 한글은 사용하지 않는다.

### 3.1 개인 실험

형식:

```text
<팀원 식별자>/exp-<세 자리 실험번호>-<실험내용>
```

예시:

```text
sdh/exp-001-eda
sdh/exp-002-logistic-baseline
sdh/exp-003-class-weight
member-b/exp-001-xgboost
member-c/exp-004-feature-selection
```

규칙:

- 팀원 식별자는 팀에서 정한 영문 소문자를 사용한다.
- 실험번호는 `001`, `002`, `003`처럼 세 자리로 통일한다.
- 실험내용은 브랜치 이름만으로 목적을 알 수 있게 작성한다.
- 여러 단어는 하이픈으로 구분한다.

피해야 할 예:

```text
test
new
experiment
sdh/test2
SDH/실험1
sdh/exp3
```

### 3.2 공용 구조 변경

형식:

```text
shared/<변경내용>
```

예시:

```text
shared/update-common-preprocessor
shared/update-result-contract
shared/add-validation-utils
shared/update-dependencies
shared/reorganize-project-structure
```

다음 영역을 변경한다면 공용 구조 변경으로 간주한다.

- `common/`
- `configs/`
- `final_pipeline/`
- `requirements.txt`
- `requirements-lock.txt`
- 루트 `README.md`
- 팀 공통 문서 및 결과 계약
- 공통 평가, 검증 또는 전처리 코드

공용 변경은 개인 실험 브랜치에 섞지 않는 것을 원칙으로 한다. 별도의 `shared/...` 브랜치와 PR로 변경 영향과 사용법을 공유한다.

### 3.3 긴급 오류 수정

공용 코드의 명확한 오류를 수정할 때는 다음 형식을 사용할 수 있다.

```text
fix/<수정내용>
```

예시:

```text
fix/incorrect-macro-f1
fix/train-test-column-order
fix/notebook-root-path
```

개인 실험 내부의 오류는 별도 `fix/...` 브랜치 대신 해당 실험 브랜치에서 수정한다.

---

## 4. 브랜치 구조

권장 구조:

```text
main
├── sdh/exp-001-eda
├── sdh/exp-002-logistic-baseline
├── member-b/exp-001-xgboost
├── member-c/exp-001-catboost
└── shared/update-common-preprocessor
```

지양하는 체인식 구조:

```text
main
└── sdh/exp-001-eda
    └── sdh/exp-002-logistic
        └── sdh/exp-003-lightgbm
```

체인식 구조에서는 앞선 실험이 병합되지 않거나 폐기될 경우 이후 실험도 영향을 받는다. PR에 이전 실험의 변경이 함께 나타나 리뷰와 병합도 어려워진다.

따라서 각 실험은 최대한 최신 `main`에서 독립적으로 생성한다.

---

## 5. 새 실험 시작

### 5.1 현재 작업 상태 확인

```bash
git status
```

커밋하지 않은 변경이 있다면 새 브랜치를 만들기 전에 해당 변경의 소유와 목적을 확인한다. 다른 실험의 변경이 남은 상태에서 임의로 브랜치를 이동하지 않는다.

### 5.2 최신 `main` 준비

```bash
git switch main
git pull origin main
```

`git pull` 중 충돌이나 오류가 발생하면 파일을 임의로 삭제하거나 덮어쓰지 말고 팀에 공유한다.

### 5.3 실험 브랜치 생성

```bash
git switch -c sdh/exp-003-lightgbm
```

현재 브랜치 확인:

```bash
git branch --show-current
```

예상 출력:

```text
sdh/exp-003-lightgbm
```

브랜치가 정확한지 확인한 후 실험을 시작한다.

---

## 6. 실험 진행 방식

### 6.1 탐색 단계

일반적인 탐색은 Jupyter Notebook에서 진행한다. 노트북 상단 설정 셀에서 다음 값을 자유롭게 변경한다.

- seed
- validation 비율 또는 fold 수
- 전처리 방식
- 모델 종류
- 하이퍼파라미터
- feature selection
- class weight
- ensemble 설정

데이터 경로와 `ID`, `SUBCLASS` 같은 고정 스키마는 루트 `configs/baseline.yaml`을 사용할 수 있다.

탐색 단계의 모든 시도를 실험별 `config.yaml`에 즉시 기록할 필요는 없다.

### 6.2 좋은 결과가 나온 경우

공유할 가치가 있는 결과가 나오면 다음 순서로 실험을 확정한다.

1. 노트북에서 사용한 설정을 확인한다.
2. 확정한 설정을 해당 실험의 `config.yaml`에 반영한다.
3. 노트북에서 변경한 전처리와 모델 코드를 Python 모듈에 반영한다.
4. 동일 조건으로 다시 실행한다.
5. 동일하거나 유사한 검증 점수가 재현되는지 확인한다.
6. 결과와 재현 정보를 `metrics.json`에 저장한다.
7. 실험 목적과 결론을 README 또는 노트북 Markdown 셀에 기록한다.
8. 노트북의 불필요한 실행 출력을 제거한다.
9. 필요한 파일만 커밋한다.

전체 흐름:

```text
Notebook 탐색
→ 좋은 결과 발견
→ config.yaml 및 Python 코드 반영
→ 동일 조건 재실행
→ 성능 재현 확인
→ metrics.json 저장
→ 커밋
→ Pull Request
```

`config.yaml`만 수정한다고 실험이 완전히 재현되는 것은 아니다. 전처리나 모델 구조를 노트북에서 변경했다면 실제 실행 코드에도 반영해야 한다.

---

## 7. 결과에 따른 공유 기준

| 실험 상태 | 코드 처리 | 공유 방식 |
|---|---|---|
| 재현되고 성능이 좋음 | PR로 `main`에 병합 | 코드, config, metrics, 실험 요약 |
| 성능은 낮지만 학습 가치가 있음 | 원칙적으로 병합하지 않음 | Notion에 아이디어, 조건, 결과, 원인 및 후속 방향 기록 |
| 단순 실행 오류 또는 설정 실수 | 공유 가치가 없으면 폐기 | 다른 팀원이 반복할 가능성이 있는 문제만 간단히 기록 |

핵심 원칙:

> 성공한 실험은 코드로 공유하고, 실패한 실험은 지식으로 공유한다.

---

## 8. 실패하거나 채택하지 않은 실험

실패한 실험 또는 baseline보다 성능이 낮은 실험은 원칙적으로 `main`에 코드를 병합하지 않는다.

대신 다른 팀원이 같은 시도를 반복하지 않거나, 기존 아이디어를 발전시킬 수 있도록 팀 Notion에 핵심 내용을 기록한다.

### 8.1 Notion에 기록할 항목

- 실험 ID
- 실험자
- 실험 날짜
- 상태
- 실험 아이디어 및 가설
- 비교한 baseline
- 변경한 내용
- 전처리 방식
- 모델과 주요 파라미터
- seed
- validation 방식
- Accuracy
- Macro F1
- baseline 대비 변화
- 채택하지 않은 이유
- 추정되는 실패 원인
- 후속으로 시도할 가치가 있는 방향
- 관련 브랜치 또는 커밋 링크
- 코드 보존 여부

### 8.2 실패 실험 기록 템플릿

```markdown
# 실험: exp-004-gene-frequency-filter

## 기본 정보

- 실험자: SDH
- 날짜: YYYY-MM-DD
- 상태: 채택하지 않음
- 관련 브랜치: `sdh/exp-004-gene-frequency-filter`

## 가설

train에서 변이 빈도가 매우 낮은 유전자를 제거하면 노이즈와 차원이 감소하여 Macro F1이 향상될 것으로 예상했다.

## 기준 실험

- 실험 ID: `exp-002-logistic-baseline`
- Accuracy: 0.8012
- Macro F1: 0.7624

## 변경사항

- train에서 변이가 5회 미만인 유전자 제거
- 나머지 전처리와 모델은 baseline과 동일
- `WT=0`, 변이값=`1`로 이진화

## 실행 조건

- Model: Logistic Regression
- Seed: 42
- Validation: Stratified holdout
- Validation ratio: 0.25

## 결과

- Accuracy: 0.7925
- Macro F1: 0.7418
- baseline 대비 Macro F1: -0.0206

## 채택하지 않은 이유

희귀 유전자를 제거한 후 전체 Macro F1이 감소했다. 특히 일부 소수 클래스의 recall이 크게 하락했다.

## 추정 원인

빈도가 낮은 유전자 중 일부가 특정 소수 암종을 구분하는 정보를 포함했을 가능성이 있다.

## 후속 아이디어

- 전체 빈도가 아니라 클래스별 변이 빈도로 필터링
- 희귀 유전자를 제거하지 않고 regularization으로 제어
- mutual information 또는 chi-square 기반 feature selection 비교
- 소수 클래스의 핵심 변이를 필터링 대상에서 제외

## 코드 보존 여부

코드는 `main`에 병합하지 않는다. 추가 분석 가능성이 있어 실험 브랜치는 당분간 유지한다.
```

### 8.3 브랜치 보존 여부

실패한 실험 브랜치를 반드시 영구 보존할 필요는 없다.

- 후속 실험 가능성이 있으면 브랜치를 일정 기간 유지하고 Notion에 링크한다.
- 코드 자체가 재사용될 가능성이 없으면 핵심 내용을 Notion에 기록한 후 브랜치를 삭제할 수 있다.
- 실행 오류나 잘못된 설정으로 발생한 결과는 지식 가치가 있을 때만 기록한다.

---

## 9. `metrics.json` 기록 기준

분류 실험에서는 최소한 다음 정보를 기록한다.

- 실험 소유자
- 실험 ID
- 모델
- seed
- validation 방식
- 전처리 요약
- 모델 파라미터
- Accuracy
- Macro F1

### 9.1 실험 폴더명과 전역 실험 ID

실험 폴더명과 `metrics.json`의 실험 ID는 역할이 다르다.

- 실험 폴더명: 개인 폴더 안에서 실험 순서와 목적을 나타낸다.
- 전역 실험 ID: `leaderboard.csv`, Notion, 결과 파일에서 팀 전체가 실험을 식별하는 공통 키다.

권장 형식:

```text
실험 폴더명: exp_<번호 3자리>_<실험 목적>
전역 실험 ID: <owner slug>-<model slug>-<번호 3자리>
```

예:

```text
experiments/member_d/exp_001_logreg_baseline/
experiment: iljun-logreg-001
```

`owner slug`는 `sdh`, `iljun`처럼 담당자를 식별할 수 있고 브랜치·Notion에서도
일관되게 사용할 수 있는 짧은 이름을 권장한다. 이미 공유된 `member-a-xgb-003`,
`member-d-logreg-001` 같은 ID는 기존 기록과 링크를 깨뜨리면서까지 소급 변경하지 않는다.

가능하면 폴더와 전역 실험 ID의 번호를 동일하게 맞춘다.

### 9.2 `validation`과 모델 파라미터

`validation`은 사람이 읽는 문자열이 아니라 구조화된 객체로 저장한다. 검증 방식에
따라 필요한 설정을 명시해야 나중에 자동 집계와 재현이 가능하다.

모델 설정 키는 `model_parameters`로 통일한다. `parameters`와
`hyperparameters`는 새 기록에서 사용하지 않는다.

예시:

```json
{
  "owner": "iljun",
  "experiment": "iljun-lightgbm-003",
  "model": "LightGBM",
  "seed": 42,
  "validation": {
    "method": "StratifiedKFold",
    "n_splits": 5,
    "shuffle": true,
    "seeds": [42]
  },
  "preprocessing": "missing->WT; WT=0; variant=1",
  "model_parameters": {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31
  },
  "accuracy": 0.8123,
  "f1_macro": 0.7845
}
```

이 프로젝트에서는 Macro F1을 주 비교 지표로 사용한다.

---

## 10. 커밋 방법

### 10.1 변경 확인

```bash
git status
git diff
```

### 10.2 필요한 파일만 추가

모든 파일을 무조건 추가하기보다 실험에 필요한 파일을 명시적으로 추가하는 것을 권장한다.

```bash
git add experiments/SDH/exp_003_lightgbm/experiment.ipynb
git add experiments/SDH/exp_003_lightgbm/config.yaml
git add experiments/SDH/exp_003_lightgbm/results/metrics.json
git add experiments/SDH/exp_003_lightgbm/README.md
```

스테이징된 내용 확인:

```bash
git diff --staged
```

### 10.3 커밋 생성

```bash
git commit -m "Add SDH LightGBM experiment"
```

다른 예시:

```bash
git commit -m "Record validation metrics for exp 003"
git commit -m "Refine mutation feature preprocessing"
```

피해야 할 메시지:

```text
update
test
수정
final
진짜최종
```

하나의 커밋에는 가능한 한 하나의 논리적 변경만 포함한다.

---

## 11. 커밋하면 안 되는 파일

다음 파일은 원칙적으로 Git에 커밋하지 않는다.

- `data/raw/`의 원본 데이터
- `data/processed/`의 가공 데이터
- 학습된 모델
- checkpoint
- `submission.csv`
- test 예측 확률
- OOF 예측 확률
- 대용량 로그
- 불필요한 노트북 실행 출력
- 개인 환경 파일
- 비밀키와 인증정보

예시:

```text
data/raw/train.csv
data/raw/test.csv
data/processed/features.parquet
model.joblib
model.pkl
checkpoint.pt
submission.csv
test_probabilities.parquet
oof_probabilities.parquet
.env
```

커밋 전에는 반드시 다음 명령으로 포함 파일을 확인한다.

```bash
git status
git diff --staged
```

---

## 12. 원격 브랜치와 Pull Request

### 12.1 최초 push

```bash
git push -u origin sdh/exp-003-lightgbm
```

이후 같은 브랜치에서 추가 커밋을 올릴 때:

```bash
git push
```

### 12.2 PR 작성

PR의 대상 브랜치는 `main`으로 설정한다.

PR 제목 예시:

```text
[SDH][EXP-003] Add LightGBM mutation baseline
```

PR 본문 예시:

```markdown
## 실험 목적

WT/variant 이진 특성에서 LightGBM의 기본 성능을 확인합니다.

## 주요 변경사항

- LightGBM 학습 코드 추가
- 실험 설정 추가
- 검증 결과 기록
- 재현 방법 문서화

## Validation

- Method: Stratified holdout
- Test size: 0.25
- Seed: 42
- Accuracy: 0.8123
- Macro F1: 0.7845

## 재현 방법

실행할 노트북 또는 명령어를 작성합니다.

## 커밋 제외 파일

- 원본 데이터
- 모델 파일
- submission
- OOF/test 확률

## 참고사항

기존 baseline 대비 변경점이나 리뷰가 필요한 내용을 작성합니다.
```

---

## 13. PR 병합 후 공유

PR이 `main`에 병합되면 작업자는 팀 채팅에 병합 사실과 주요 영향을 공유한다.

개인 실험 병합 예시:

```text
SDH exp-003-lightgbm PR이 main에 병합되었습니다.

새로운 작업을 시작하기 전에 main을 pull해서 최신화해 주세요.

주요 변경:
- LightGBM 실험 추가
- Macro F1 결과 기록
- 공용 전처리 코드 변경 없음
```

공용 구조 변경 예시:

```text
shared/update-common-preprocessor PR이 main에 병합되었습니다.

common/starter_preprocess.py의 인터페이스가 변경되었습니다.
새로운 작업을 시작하기 전에 main을 pull해 주세요.

기존 실험 코드에서는 transform 호출 방법을 확인해야 합니다.
```

공용 변경이라면 영향받는 파일과 사용 방법을 반드시 함께 적는다.

---

## 14. 다른 PR 병합 후 최신화

새로운 작업을 시작하기 전에:

```bash
git switch main
git pull origin main
```

최신 `main`에서 새 실험 브랜치를 생성한다.

```bash
git switch -c sdh/exp-004-feature-selection
```

---

## 15. 작업 중인 브랜치에 최신 `main` 반영

진행 중인 실험에 최신 `main` 변경이 필요한 경우:

```bash
git switch main
git pull origin main
git switch sdh/exp-003-lightgbm
git merge main
```

충돌 확인:

```bash
git status
```

충돌 해결 후:

```bash
git add <충돌을 해결한 파일>
git commit
```

팀 기본 방식은 이해하기 쉬운 `merge main`으로 통일한다. `rebase`는 Git 사용에 익숙하고 팀 합의가 있을 때만 사용한다.

---

## 16. 체인식 브랜치의 예외

다음 실험이 아직 병합되지 않은 이전 실험의 코드에 반드시 의존한다면 일시적으로 이전 실험 브랜치에서 분기할 수 있다.

```text
main
└── sdh/exp-002-logistic
    └── sdh/exp-003-logistic-calibration
```

이 방식은 예외적으로만 사용한다.

- exp3 PR에 exp2의 변경까지 함께 나타날 수 있다.
- exp2가 변경되면 exp3에서도 충돌이 발생할 수 있다.
- exp2가 폐기되면 exp3의 기반을 다시 정리해야 한다.
- 가능하면 exp2를 먼저 `main`에 병합한 뒤 exp3를 생성한다.

체인식 분기가 필요하다면 팀 채팅과 PR에 의존관계를 명시한다.

```text
exp-003은 아직 병합되지 않은 exp-002의 전처리 코드에 의존합니다.
exp-002가 먼저 main에 병합되어야 합니다.
```

---

## 17. 개인 실험 중 공용 변경이 필요한 경우

개인 실험 중 공용 코드 변경이 필요해졌다면 다음처럼 분리한다.

1. `shared/...` 브랜치에서 공용 변경 작업
2. 공용 변경 PR 리뷰 및 `main` 병합
3. 로컬 `main` 최신화
4. 최신 `main`에서 개인 실험 브랜치 생성
5. 이미 실험 중이라면 해당 브랜치에 최신 `main` 병합

예시:

```text
shared/add-feature-selector
sdh/exp-005-feature-selection
```

개인 실험 PR 하나에 대규모 공용 구조 변경과 모델 실험을 함께 포함하지 않는다.

---

## 18. 병합된 브랜치 정리

PR이 병합된 뒤 원격 브랜치를 삭제할 수 있다.

```bash
git push origin --delete sdh/exp-003-lightgbm
```

로컬 브랜치 삭제:

```bash
git switch main
git branch -d sdh/exp-003-lightgbm
```

`git branch -D`는 병합되지 않은 변경도 강제로 삭제할 수 있으므로 일반적으로 사용하지 않는다.

---

## 19. 작업 시작 전 체크리스트

- [ ] `git status`로 현재 변경사항을 확인했다.
- [ ] `main`으로 이동했다.
- [ ] `git pull origin main`으로 최신화했다.
- [ ] 최신 `main`에서 실험 브랜치를 생성했다.
- [ ] 브랜치 이름이 규칙에 맞는다.
- [ ] `git branch --show-current`로 현재 브랜치를 확인했다.

---

## 20. 커밋 전 체크리스트

- [ ] 다른 팀원의 실험 디렉터리를 수정하지 않았다.
- [ ] 공용 파일 변경이 개인 실험에 불필요하게 섞이지 않았다.
- [ ] 원본 또는 가공 데이터가 포함되지 않았다.
- [ ] 모델, checkpoint, submission이 포함되지 않았다.
- [ ] OOF/test 확률 파일이 포함되지 않았다.
- [ ] 노트북의 불필요한 실행 출력을 제거했다.
- [ ] 좋은 결과라면 `metrics.json`에 재현 정보를 기록했다.
- [ ] `git diff --staged`로 커밋 내용을 확인했다.

---

## 21. PR 전 체크리스트

- [ ] 코드와 노트북이 오류 없이 실행된다.
- [ ] 사용한 seed와 validation 방식이 기록되어 있다.
- [ ] Accuracy와 Macro F1이 기록되어 있다.
- [ ] 전처리 방식과 모델 파라미터가 기록되어 있다.
- [ ] `config.yaml`과 실제 확정 실행 설정이 일치한다.
- [ ] 실험 재현 방법이 작성되어 있다.
- [ ] PR 대상 브랜치가 `main`이다.
- [ ] 병합 후 팀에 공유할 내용을 준비했다.

---

## 22. 전체 작업 흐름

```text
1. git status
2. git switch main
3. git pull origin main
4. git switch -c <팀원>/exp-<번호>-<내용>
5. Notebook에서 실험
6. 결과 평가

성공한 실험:
7. config.yaml과 Python 코드에 확정 설정 반영
8. 재실행 및 성능 재현 확인
9. metrics.json 저장
10. 필요한 파일만 git add
11. git diff --staged
12. git commit
13. git push
14. Pull Request
15. main 병합
16. 팀에 병합 내용 공유

실패하거나 채택하지 않은 실험:
7. 아이디어, 조건, 결과 및 실패 원인을 Notion에 기록
8. 후속 실험 가능성 기록
9. 필요하면 브랜치 링크 공유
10. main에는 원칙적으로 병합하지 않음
```

---

## 23. 최종 요약

> 개인 실험은 최신 `main`에서 독립적인 브랜치로 시작한다.

> 이전 실험 브랜치에서 다음 실험 브랜치를 연속으로 생성하지 않는다.

> 좋은 결과는 재현 가능한 코드, 설정 및 지표와 함께 `main`에 반영한다.

> 실패한 결과는 다른 팀원의 중복 작업을 방지하고 후속 아이디어를 발전시킬 수 있도록 Notion에 지식으로 공유한다.

> PR이 `main`에 병합되면 반드시 팀에 공유하고, 다른 팀원은 새 작업 전에 `main`을 최신화한다.
