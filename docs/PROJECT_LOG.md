# Project Decision Log

프로젝트 전체에 영향을 주는 결정과 검증 결과를 기록합니다. 개인 실험의 세부 과정은 각 팀원의 노트북이나 실험 폴더에 기록하고, 이 문서에는 모두가 알아야 하는 내용만 남깁니다.

## 2026-07-30 — 공용 전처리 벤치마크 도입

### 결정

- 전처리 성능 비교는 팀 공용 `StratifiedKFold-5`와 고정 모델로 수행한다.
- 기본 CV seed는 42이며, 최종 후보는 42, 52, 62에서 반복 검증한다.
- 1차 모델은 고정된 Logistic Regression, 2차 모델은 고정된 LightGBM을 사용한다.
- 주 비교 지표는 전체 OOF 예측의 Macro F1이다.
- 팀원은 sklearn 호환 Transformer 또는 전처리 `Pipeline`만 구현하며, fold·모델·평가·OOF 생성은 공용 코드가 담당한다.
- feature selection과 통계 학습은 각 fold의 train 부분에서만 수행한다.

### 영향

- 공용 비교 결과는 `common.preprocessing_benchmark.run_preprocessing_benchmark`에 `preprocessor` 객체를 전달해 생성한다.
- 탐색 단계의 홀드아웃 결과는 후보 제거에 사용할 수 있지만, 전처리 순위와 최종 후보 선정에는 5-fold OOF 결과를 사용한다.
- 좋은 전처리만 동일 fold의 LightGBM으로 2차 검증한다.

### 검증

- 합성 다중 클래스 데이터에서 fold별 전처리 fit, OOF 행 완전성, 클래스 확률 정렬 및 지표 생성을 검사한다.
- 실제 데이터에서 fold train 기준 활성 유전자 선택, WT 이진화, `log1p(TMB)` 전처리로 5-fold Logistic Regression을 실행했다.
- seed 42의 OOF Macro F1은 0.36236이었으며 기존 기준값 0.36296과 유사했다.
- fold마다 train 부분에서만 활성 유전자를 선택하여 feature 수가 4,227~4,231개로 달라지는 것을 확인했다.

## 2026-07-30 — 초기 협업 구조 확정

### 결정

- Python 버전은 3.12로 통일한다.
- 초보 팀원이 바로 시작할 수 있도록 JupyterLab을 기본 실험 환경으로 사용한다.
- 팀원 A, B, C, D는 각각 `experiments/member_a`부터 `member_d`까지 자신의 공간에서 독립적으로 실험한다.
- 개인 폴더 내부 구조는 강제하지 않는다. 제공된 `exp_001_baseline`은 선택적으로 참고할 수 있는 예시다.
- 원본 데이터는 루트의 `data/raw/`에서 공동으로 읽고, 팀원 폴더마다 데이터를 복사하지 않는다.
- seed, 검증 비율, fold, 전처리, 모델 파라미터는 고정 규칙이 아니라 각 실험에서 변경할 수 있는 변수로 취급한다.
- 기본 비교 지표는 Macro F1이며 Accuracy도 함께 기록한다.
- 모델, 제출 파일, 확률 파일처럼 큰 산출물은 Git에 올리지 않고, 재현에 필요한 설정과 `metrics.json` 같은 작은 기록만 공유한다.
- 사후 확률 앙상블과 OOF stacking을 위해 ID, 클래스 순서, fold, 실제 정답 등 공유 산출물 형식을 맞춘다.
- 향후 딥러닝 실험도 개인 폴더에 자유롭게 추가하고, 최종 채택된 처리만 `final_pipeline/`에 반영한다.

### 구현

- 네 팀원용 Quick Start 노트북을 추가했다.
- README를 JupyterLab 중심의 초보자 안내서로 개편했다.
- 선택형 CLI baseline과 공통 모델 인터페이스를 유지했다.
- Windows, macOS, Linux/WSL 개발 파일과 데이터·모델·제출물·확률 산출물을 `.gitignore`에서 제외하도록 정리했다.
- 직접 의존성은 `requirements.txt`, 검증된 WSL 환경은 `requirements-lock.txt`로 관리한다.
- Codex가 저장소 작업 시 따를 규칙을 루트 `AGENTS.md`에 기록했다.

### 검증

- 실제 데이터의 train/test/submission 열 구성, ID 중복 여부, test와 submission의 ID 및 순서를 확인했다.
- 네 Quick Start 노트북의 JSON 형식, 소유자별 경로, 비어 있는 출력 상태를 확인했다.
- 팀원 A Quick Start 노트북을 실제 데이터로 끝까지 실행해 학습, Macro F1 계산, 제출 생성 과정을 확인했다.
- 선택형 CLI baseline의 학습과 저장 모델 기반 추론을 확인했다.
- WSL Python 3.12 가상환경에서 패키지 의존성 충돌이 없음을 확인했다.

## 기록을 추가하는 방법

공통 규칙이나 구조가 바뀔 때 아래 형식으로 문서 위쪽에 새 항목을 추가합니다.

```markdown
## YYYY-MM-DD — 변경 제목

### 결정

- 무엇을 왜 바꿨는지

### 영향

- 팀원이 새로 지켜야 하는 내용

### 검증

- 실제로 확인한 실행 또는 검사 결과
```
