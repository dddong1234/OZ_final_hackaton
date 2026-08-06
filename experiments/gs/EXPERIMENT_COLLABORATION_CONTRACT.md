# GS 실험 협업·검증 계약

## 1. Git 산출물 관리

다음 대용량 산출물은 Git에 커밋하지 않는다. 실행 후 로컬에 보관하고, 필요하면 별도 공유 저장소를 사용한다.

- 학습된 모델
- 전체 OOF 확률
- test 확률
- 제출 CSV
- checkpoint
- notebook 대용량 출력

Git에는 다음 경량 정보만 남긴다.

- seed별 Macro F1
- fold별 Macro F1
- Accuracy
- 클래스별 F1
- 피처 수
- 수렴 경고 수
- 누수 감사 결과
- 채택·기각 판정

기존에 Git 추적 이력이 있는 대용량 파일은 이 계약 도입 이전의 역사적 산출물로 본다. 새 실험에서는 `.gitignore`를 적용하며, 과거 히스토리 재작성은 팀 합의 없이는 수행하지 않는다.

## 2. 공통 검증 계약

팀원의 실험 결과를 공정하게 비교하기 위해 다음 조건을 통일한다.

- 주 평가 지표: Macro F1
- 보조 지표: Accuracy
- 검증: Stratified 5-fold
- 기본 seeds: `42 / 777 / 2024`
- Logistic Regression: `C=0.07`, `max_iter=2000`
- 학습형 전처리는 fold-train에서만 fit
- validation에는 학습된 변환만 적용
- test 기반 통계·vocabulary·threshold·피처 선택 금지
- 동일한 ID 및 행 순서 사용
- 동일한 클래스와 확률 열 순서 사용
- OOF 확률과 fold 정보를 기준으로 모델 비교

이 계약은 서로 다른 데이터 분할이나 변환 기준이 만든 유불리를 줄이고, 동일 조건에서 아이디어가 실제 개선을 만드는지 평가하기 위한 것이다.

## 3. OOF 산출물 계약

앙상블이나 독립 검증에 사용하는 로컬 OOF 파일은 아래 열을 포함한다.

```text
ID
SUBCLASS
seed
fold
prob_ACC
prob_BLCA
...
prob_UCEC
```

필수 확인 항목:

- OOF 행이 자기 fold의 학습에 사용되지 않았는가
- 원본 ID 순서가 유지되는가
- 클래스 열 순서가 명시되어 있는가
- 각 행의 확률합이 1인가
- NaN 또는 무한대가 없는가
- seed별 결과가 평균 전에 별도로 보존되는가

test 확률은 다음 형식을 사용한다.

```text
ID
prob_ACC
prob_BLCA
...
prob_UCEC
```

test 확률은 후보·가중치·전처리 규칙이 모두 train-only 검증에서 확정된 뒤에만 생성한다.

## 4. 안전성 감사

모든 실행기는 가능한 범위에서 다음을 기록한다.

- `leakage_check=True`
- `nan_as_mutation_count=0`
- 수렴 경고 수
- seed·fold별 점수
- 피처 수와 실행 시간

`WT`, 빈 문자열, NaN은 mutation event로 만들지 않는다. train과 test를 결합해 통계·인코딩·피처 선택을 수행하지 않는다.
