# AI Modeling Hackathon

팀원 A, B, C, D가 Jupyter Notebook에서 자유롭게 실험하고, 좋은 결과만 공통 형식으로
공유하기 위한 저장소입니다.

기본 작업 도구는 JupyterLab입니다. 터미널은 저장소 받기, 가상환경 만들기,
JupyterLab 실행과 Git 작업에만 사용합니다.

## 반드시 지킬 규칙

1. `data/raw/`의 원본 파일을 수정하지 않습니다.
2. 다른 팀원의 `experiments/member_x/` 폴더를 수정하지 않습니다.
3. JupyterLab은 저장소 루트에서 실행합니다.
4. 공용 파일을 변경하기 전 팀에 알립니다.
5. 모델, checkpoint, submission과 확률 파일은 Git에 올리지 않습니다.
6. 공유할 결과에는 실험 ID, seed, 검증 방식과 Macro F1을 기록합니다.

개인 폴더 내부의 노트북 이름, 폴더 구조, 모델과 프레임워크는 자유입니다.

---

## 1. 처음 한 번만 준비하기

### 1-1. 저장소 받기

터미널을 열고 저장소를 받습니다.

```bash
git clone https://github.com/dddong1234/OZ_fianl_hackaton.git
cd OZ_fianl_hackaton
```

현재 위치의 파일을 확인합니다.

WSL/Linux/macOS:

```bash
ls
```

Windows 명령 프롬프트:

```bat
dir
```

`README.md`, `experiments`, `data`, `requirements.txt`가 보이면 저장소 루트에 있는
것입니다.

앞으로 별도 설명이 없는 터미널 명령은 이 위치에서 실행합니다.

### 1-2. Python 확인

팀 공통 버전은 Python 3.12 계열입니다.

```bash
python --version
```

환경에 따라 다음 명령을 사용합니다.

```bash
python3 --version
```

Windows:

```powershell
py -3.12 --version
```

출력에 `Python 3.12`가 포함되면 됩니다.

### 1-3. 가상환경과 패키지 설치

가상환경은 프로젝트 전용 Python 환경입니다. 저장소 루트에 `.venv`라는 이름으로
만듭니다.

#### WSL 또는 Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip check
```

#### Windows PowerShell

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

#### Windows 명령 프롬프트

```bat
py -3.12 -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

#### macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

`No broken requirements found.`가 나오면 정상입니다.

가상환경 확인:

```bash
python -c "import sys; print(sys.executable)"
```

출력 경로에 `.venv`가 포함돼야 합니다.

---

## 2. 데이터 준비하기

다음 파일을 `data/raw/`에 넣습니다.

```text
data/raw/
├─ train.csv
├─ test.csv
└─ sample_submission.csv
```

파일명을 변경하지 않습니다.

데이터 확인:

WSL/Linux/macOS:

```bash
ls -lh data/raw
```

Windows PowerShell:

```powershell
Get-ChildItem data\raw
```

주요 컬럼:

```text
ID        샘플 식별자
SUBCLASS  예측 대상
```

`SUBCLASS`는 train에만 있습니다. 원본 데이터는 Git에서 자동으로 제외됩니다.

---

## 3. JupyterLab 실행하기

저장소 루트에서 가상환경을 활성화한 후 실행합니다.

WSL/Linux/macOS:

```bash
source .venv/bin/activate
jupyter lab
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
jupyter lab
```

Windows 명령 프롬프트:

```bat
.\.venv\Scripts\activate.bat
jupyter lab
```

브라우저가 자동으로 열리지 않으면 터미널에 표시된 `http://localhost:...` 주소를
브라우저에 붙여 넣습니다.

### 커널 선택

노트북 오른쪽 위의 커널 이름을 확인합니다. `.venv`의 Python이 아니라면 다음
순서로 변경합니다.

```text
Kernel
→ Change Kernel
→ Python 3 또는 현재 .venv의 Python 선택
```

확실하지 않다면 노트북 셀에서 확인합니다.

```python
import sys

print(sys.executable)
```

경로에 `.venv`가 포함되면 정상입니다.

Jupyter에 가상환경 커널이 보이지 않으면 터미널에서 다음 명령을 한 번 실행합니다.

```bash
python -m ipykernel install --user \
  --name oz-hackathon \
  --display-name "Python 3.12 (OZ Hackathon)"
```

그 후 JupyterLab을 다시 열고 `Python 3.12 (OZ Hackathon)` 커널을 선택합니다.

---

## 4. Quick Start 노트북 실행하기

각 팀원 폴더에 입문용 노트북이 있습니다.

```text
experiments/
├─ member_a/notebooks/00_quick_start.ipynb
├─ member_b/notebooks/00_quick_start.ipynb
├─ member_c/notebooks/00_quick_start.ipynb
└─ member_d/notebooks/00_quick_start.ipynb
```

자신의 팀원 폴더에 있는 노트북을 엽니다.

JupyterLab 메뉴에서 다음을 선택합니다.

```text
Run
→ Run All Cells
```

또는 셀을 하나씩 선택하고 `Shift + Enter`를 누릅니다.

Quick Start 노트북은 다음 작업을 수행합니다.

```text
프로젝트 경로 찾기
→ train/test/submission 읽기
→ 데이터 크기와 클래스 확인
→ train/validation 분리
→ WT/변이 이진 전처리
→ Logistic Regression 학습
→ Accuracy와 Macro F1 계산
→ 전체 train 재학습
→ submission과 모델 저장
```

팀원 A의 결과는 다음 위치에 저장됩니다.

```text
experiments/member_a/results/member-a-notebook-baseline-001/
├─ metrics.json
├─ model.joblib
└─ submission.csv
```

| 파일 | 의미 | Git |
|---|---|---|
| `metrics.json` | 실험 정보와 검증 점수 | 공유 가능 |
| `model.joblib` | 학습 모델 | 제외 |
| `submission.csv` | 대회 제출 파일 | 제외 |

터미널에서 Quick Start를 실행할 필요는 없습니다. JupyterLab에서 셀을 실행하면 됩니다.

---

## 5. 내 노트북 실험 만들기

개인 노트북 구조는 자유입니다. 다음은 권장 예시일 뿐입니다.

```text
experiments/member_a/notebooks/
├─ 00_quick_start.ipynb
├─ 01_eda.ipynb
├─ 02_preprocessing.ipynb
├─ 03_random_forest.ipynb
├─ 04_xgboost.ipynb
└─ 05_ensemble.ipynb
```

새 노트북을 만드는 방법:

1. JupyterLab 왼쪽 파일 목록에서 자신의 `notebooks/` 폴더를 엽니다.
2. `+` 버튼을 누릅니다.
3. Notebook의 Python 커널을 선택합니다.
4. 파일명을 실험 목적에 맞게 변경합니다.

다른 팀원의 폴더에는 노트북을 만들지 않습니다.

### 첫 번째 셀에 실험 정보 적기

Markdown 셀 예시:

```markdown
# XGBoost mutation count

- Owner: member_a
- Experiment ID: member-a-xgb-003
- Seed: 42
- Validation: StratifiedKFold 5
- 목적: 변이 개수 피처가 Macro F1에 미치는 영향 확인
```

코드 셀에도 같은 설정을 변수로 둡니다.

```python
MEMBER = "member_a"
EXPERIMENT_ID = "member-a-xgb-003"
SEED = 42
```

실험 ID는 다른 실험과 겹치지 않게 작성합니다.

---

## 6. 노트북에서 프로젝트 경로 찾기

JupyterLab을 저장소 루트에서 실행하면 보통 `Path.cwd()`가 저장소 루트입니다.
노트북 위치나 실행 방식이 달라져도 안전하게 찾으려면 다음 코드를 사용합니다.

```python
from pathlib import Path


def find_project_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "configs" / "baseline.yaml").exists():
            return path
    raise FileNotFoundError(
        "저장소 루트를 찾지 못했습니다. "
        "JupyterLab을 저장소 루트에서 실행하세요."
    )


ROOT = find_project_root(Path.cwd())
print(ROOT)
```

개인 PC의 절대경로를 노트북에 직접 쓰지 않습니다.

피해야 할 예:

```python
ROOT = Path(r"C:\Users\내이름\Desktop\project")
```

---

## 7. 노트북에서 데이터 읽기

```python
import pandas as pd


train = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
test = pd.read_csv(ROOT / "data" / "raw" / "test.csv")
submission = pd.read_csv(
    ROOT / "data" / "raw" / "sample_submission.csv"
)

print("train:", train.shape)
print("test:", test.shape)
display(train.head())
```

기본 확인:

```python
print(train["SUBCLASS"].nunique())
display(train["SUBCLASS"].value_counts())
print("train missing:", train.isna().sum().sum())
print("test missing:", test.isna().sum().sum())
```

수천 개 컬럼 전체를 출력하면 브라우저가 느려질 수 있습니다. `head()`나 필요한
컬럼만 출력합니다.

---

## 8. 노트북에서 train/validation 나누기

성능 비교를 위해 검증 데이터를 분리합니다.

```python
from sklearn.model_selection import train_test_split


target = "SUBCLASS"
id_column = "ID"

train_part, valid_part = train_test_split(
    train,
    test_size=0.25,
    random_state=SEED,
    stratify=train[target],
)

print(train_part.shape)
print(valid_part.shape)
```

`stratify=train[target]`는 train과 validation의 클래스 비율을 비슷하게 유지합니다.

중요한 규칙:

- 전처리 규칙은 `train_part`로만 학습합니다.
- validation 성능을 보고 test 정답을 추측하지 않습니다.
- 모델 비교 시 같은 seed와 validation을 사용하는 것이 좋습니다.

---

## 9. 노트북에서 전처리하기

기본 전처리는 `WT`를 0, 변이가 있으면 1로 바꿉니다.

```python
feature_columns = [
    column
    for column in train.columns
    if column not in {target, id_column}
]


def transform(df):
    return (
        df[feature_columns]
        .fillna("WT")
        .ne("WT")
        .astype("int8")
    )


x_train = transform(train_part)
x_valid = transform(valid_part)
x_test = transform(test)
```

전처리 결과 확인:

```python
print(x_train.shape, x_valid.shape, x_test.shape)
print("NaN:", x_train.isna().sum().sum())
```

확인할 내용:

- ID와 SUBCLASS가 피처에 포함되지 않았는가?
- train/validation/test 컬럼 수와 순서가 같은가?
- NaN이나 무한대가 남아 있지 않은가?
- validation/test 정보를 이용해 전처리 기준을 학습하지 않았는가?

중앙값, 스케일러, 피처 선택처럼 학습이 필요한 전처리는 `train_part`에서만 fit하고
validation/test에는 transform만 적용합니다.

---

## 10. 노트북에서 모델 학습하고 Macro F1 확인하기

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


model = LogisticRegression(
    max_iter=1000,
    random_state=SEED,
)
model.fit(x_train, train_part[target])

valid_pred = model.predict(x_valid)

accuracy = accuracy_score(valid_part[target], valid_pred)
f1_macro = f1_score(
    valid_part[target],
    valid_pred,
    average="macro",
)

print(f"Accuracy: {accuracy:.6f}")
print(f"Macro F1: {f1_macro:.6f}")
```

대회 비교에서는 Macro F1을 우선 확인합니다.

다른 모델도 같은 방식으로 실험할 수 있습니다.

```python
from sklearn.ensemble import RandomForestClassifier


model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    random_state=SEED,
    n_jobs=-1,
    class_weight="balanced",
)
```

XGBoost, LightGBM, CatBoost도 공통 requirements에 포함돼 있습니다.

---

## 11. 결과 저장하기

결과는 자신의 폴더 안에 저장합니다.

```python
from pathlib import Path
import json
import joblib


RESULT_DIR = (
    ROOT
    / "experiments"
    / MEMBER
    / "results"
    / EXPERIMENT_ID
)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
```

### metrics 저장

```python
metrics = {
    "experiment": EXPERIMENT_ID,
    "owner": MEMBER,
    "model": type(model).__name__,
    "seed": SEED,
    "validation": "StratifiedHoldout(test_size=0.25)",
    "accuracy": float(accuracy),
    "f1_macro": float(f1_macro),
    "description": "실험 설명 작성",
}

(RESULT_DIR / "metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
```

### 전체 train으로 최종 학습 후 submission 저장

validation 점수를 확인한 후 제출할 모델을 전체 train으로 다시 학습합니다.

```python
final_model = LogisticRegression(
    max_iter=1000,
    random_state=SEED,
)
final_model.fit(transform(train), train[target])

test_pred = final_model.predict(transform(test))

submission[id_column] = test[id_column].values
submission[target] = test_pred
submission.to_csv(
    RESULT_DIR / "submission.csv",
    index=False,
)

joblib.dump(
    final_model,
    RESULT_DIR / "model.joblib",
)
```

파일 확인:

```python
print(list(RESULT_DIR.iterdir()))
display(submission.head())
```

---

## 12. Seed와 검증 방식 바꾸기

Seed만 변경:

```python
SEED = 2026
```

여러 seed를 실험하려면 반복문으로 학습합니다.

```python
seeds = [42, 52, 62, 72, 82]
scores = []

for seed in seeds:
    # seed마다 split, 학습, 평가
    # scores에 Macro F1 저장
    pass
```

처음에는 한 seed로 실험하고, 좋은 모델을 고른 뒤 여러 seed 앙상블을 시도하는 것을
권장합니다.

OOF가 필요하면 `StratifiedKFold`를 사용합니다.

```python
from sklearn.model_selection import StratifiedKFold


skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)
```

---

## 13. 결과를 팀에 공유하기

모든 실험을 공유할 필요는 없습니다. 비교할 가치가 있거나 최종 후보로 검토할 실험만
공유합니다.

### 점수 공유

다음 내용을 전달합니다.

```text
Experiment ID
Owner
모델
Seed 또는 fold
Validation 방식
Accuracy
Macro F1
전처리·피처 설명
```

좋은 결과는 `leaderboard.csv`에 추가할 수 있습니다.

```csv
experiment,owner,model,seed,validation,accuracy,f1_macro,description
member-a-xgb-003,member_a,XGBoost,42,StratifiedKFold-5,0.42,0.38,WT binary
```

### 사후 확률 앙상블에 참여

앙상블에 사용할 실험만 `test_probabilities.parquet`을 준비합니다.

```python
test_probability = pd.DataFrame(
    final_model.predict_proba(transform(test)),
    columns=final_model.classes_,
)
test_probability.insert(0, "ID", test["ID"].values)
test_probability.to_parquet(
    RESULT_DIR / "test_probabilities.parquet",
    index=False,
)
```

규칙:

- ID 중복과 누락이 없어야 합니다.
- 앙상블 전에 ID로 정렬합니다.
- 클래스 컬럼명은 실제 SUBCLASS 문자열과 같아야 합니다.
- 클래스명을 기준으로 컬럼을 정렬합니다.
- 각 행의 확률 합은 오차 `1e-6` 이내에서 1이어야 합니다.

### OOF stacking에 참여

OOF stacking용 파일:

```text
ID + fold + true_label + 26개 클래스 확률
```

각 행은 해당 행을 학습에 사용하지 않은 fold 모델로 예측해야 합니다. 모든 train ID가
정확히 한 번 포함돼야 하며 중복과 누락이 없어야 합니다.

모델, submission과 확률 파일은 Git에 올리지 않고 팀 공용 Drive 등을 사용합니다.

---

## 14. 딥러닝 노트북 실험

PyTorch/TensorFlow도 자신의 폴더에서 자유롭게 사용할 수 있습니다.

간단한 실험은 하나의 노트북에서 진행할 수 있습니다.

```text
experiments/member_a/notebooks/10_pytorch_mlp.ipynb
```

코드가 길어지면 노트북과 Python 파일을 분리합니다.

```text
experiments/member_a/dl_mlp_001/
├─ experiment.ipynb
├─ dataset.py
├─ model.py
├─ train.py
├─ inference.py
└─ results/
```

PyTorch/TensorFlow는 CUDA 환경에 따라 설치 방식이 다르므로 공통 requirements에는
포함하지 않습니다. 도입 전에 팀이 다음을 정합니다.

- 프레임워크 버전
- CUDA 버전
- checkpoint 형식
- seed와 deterministic 설정
- OOF/test 확률 형식

`.pt`, `.pth`, `.ckpt`, `.h5`, `.keras`, `.safetensors` 파일은 Git에서 자동으로
제외됩니다.

---

## 15. CLI 베이스라인은 선택 사항

노트북 대신 Python 모듈 방식으로 실행하고 싶은 팀원만 사용합니다.

EDA:

```bash
python -m experiments.member_a.notebooks.run_eda
```

학습·검증·테스트 예측:

```bash
python -m experiments.member_a.exp_001_baseline.training.run
```

저장 모델 추론:

```bash
python -m experiments.member_a.exp_001_baseline.inference
```

팀원 B~D는 `member_a`를 자신의 폴더명으로 바꿉니다.

CLI 베이스라인은 참고용이며 개인 노트북에 이 구조를 강제하지 않습니다.

---

## 16. 노트북을 Git에 올릴 때

노트북 자체는 Git에 올릴 수 있습니다. 다만 셀 출력에 대용량 데이터나 이미지가
포함되면 파일이 매우 커집니다.

commit 전 확인:

- DataFrame 전체를 출력하지 않았는가?
- 긴 로그가 남아 있지 않은가?
- 개인 경로나 비밀값이 포함되지 않았는가?
- 다른 사람이 위에서 아래로 다시 실행할 수 있는가?
- 필요한 seed와 실험 설명이 적혀 있는가?

출력을 모두 지우는 방법:

JupyterLab 메뉴:

```text
Edit
→ Clear Outputs of All Cells
```

또는 터미널:

```bash
jupyter nbconvert \
  --clear-output \
  --inplace \
  experiments/member_a/notebooks/내노트북.ipynb
```

Quick Start 노트북도 commit 시에는 출력이 없는 상태를 유지합니다.

---

## 17. Git으로 협업하기

### 작업 시작

```bash
git switch main
git pull
git switch -c member-a/exp-xgboost
```

브랜치 예시:

```text
member-a/exp-xgboost
member-b/eda-target
member-c/fix-preprocessing
```

### 변경 확인

```bash
git status
```

자신의 노트북과 공유할 `metrics.json`만 포함되는지 확인합니다. 원본 데이터, 모델,
submission과 확률 파일은 표시되지 않아야 합니다.

### commit과 push

```bash
git add experiments/member_a
git commit -m "exp(member-a): add xgboost notebook"
git push -u origin member-a/exp-xgboost
```

`member_a`는 자신의 폴더명으로 바꿉니다.

공용 파일을 변경했다면 PR 설명에 이유를 적습니다.

PR에 적을 내용:

```text
실험 ID
변경 내용
Seed와 validation
Macro F1
공용 파일 변경 여부
```

---

## 18. 공용 폴더의 역할

| 폴더 | 용도 |
|---|---|
| `common/` | 선택형 CLI 베이스라인 공통 코드 |
| `configs/` | 데이터 경로와 비교용 validation 기본값 |
| `candidates/` | 재현과 누수 검토를 통과한 후보 |
| `final_pipeline/` | 최종 후보 결정 후 공동 구성 |
| `outputs/` | 팀원이 함께 확인할 임시 산출물 |

`common/`, `configs/`, 환경 파일과 README를 변경할 때는 팀에 먼저 알립니다.

---

## 19. 자주 발생하는 문제

### `jupyter: command not found`

가상환경을 활성화하고 패키지를 설치합니다.

WSL/Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

Windows/macOS:

```bash
python -m pip install -r requirements.txt
```

### 노트북에서 패키지를 찾지 못함

노트북 셀에서 확인합니다.

```python
import sys

print(sys.executable)
```

`.venv`가 포함되지 않으면 JupyterLab 커널을 변경합니다.

### 데이터를 찾지 못함

다음 위치를 확인합니다.

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

JupyterLab을 저장소 루트에서 실행했는지도 확인합니다.

### 브라우저가 느려짐

수천 개 컬럼이나 긴 로그를 한 셀에 출력하지 않습니다.

```python
display(train.head())
display(train[["ID", "SUBCLASS"]].head())
```

### 커널을 재시작했더니 변수가 없어짐

노트북 변수는 커널 메모리에만 존재합니다. 위에서부터 셀을 다시 실행합니다.

```text
Run
→ Run All Cells
```

다른 사람이 재현할 수 있도록 셀은 위에서 아래 순서로 실행 가능하게 작성합니다.

### PowerShell에서 가상환경 실행이 차단됨

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```
