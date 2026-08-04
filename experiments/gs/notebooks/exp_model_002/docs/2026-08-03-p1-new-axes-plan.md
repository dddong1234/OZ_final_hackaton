# P1 고정 신규 축 실험 계획

## 공통 계약

- 데이터는 `data/raw/train.csv`만 학습·통계·선택에 사용한다. test는 실행기의 제출 단계가 아닌 이상 읽지 않는다.
- WT, 빈 문자열, NaN은 mutation event가 아니다. 모든 실행 결과에 `nan_as_mutation_count=0`을 기록한다.
- 외부 데이터·사전학습·팀원 파일 import는 사용하지 않는다. 새 `common/p1_core.py`에 필요한 파서와 P1을 독립 구현한다.
- 고정 검증은 Stratified 5-fold, 개발 screen은 seed 42 하나다. 통과 후보만 42/777/2024 3-seed로 별도 확정한다.
- 기본 분류기는 `LogisticRegression(lbfgs, C=0.07, max_iter=2000, class_weight='balanced')`다.

## 실행 순서

1. `exp-p1-residual-audit-01.ipynb`: P1의 잔여 오류 margin, entropy, 주요 혼동쌍, profile 충돌 여부를 감사한다.
2. `exp-dense-enrichment-lgbm-01.ipynb`: P1의 26개 enrichment 점수와 19개 행 구조 피처만으로 얕은 LightGBM을 학습한다. H1 단독만 screen하며 H2 blend는 자동 실행하지 않는다.
3. `exp-comutation-enrichment-01.ipynb`: fold-train recurrent gene pair의 26차원 class-enrichment를 screen한다. C1만 실행하고 C2 결합은 게이트 통과 뒤 생성한다.
4. `exp-empirical-bayes-enrichment-01.ipynb`: P1과 토큰 정의를 동일하게 두고 Beta-Binomial posterior log-odds로 추정만 대체한다.
5. `exp-p1-ovr-01.ipynb`: P1 입력 그대로 multinomial LR과 OVR LR을 비교한다.

## Screen 통과 게이트

- seed 42에서 P1 대비 Macro F1 `+0.003` 이상
- 5개 fold 중 4개 이상 상승
- 수렴 경고 0, 유한 확률, leakage 검사 통과, NaN mutation 0

각 screen은 새 피처·파라미터·blend 비율을 추가 탐색하지 않는다. 통과 후보만 별도 3-seed 확정 검증으로 확장한다.
