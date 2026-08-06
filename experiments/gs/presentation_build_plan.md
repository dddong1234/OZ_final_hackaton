# EDA·전처리 발표 HTML/PPTX 구현 계획

**목표:** 팀 발표 템플릿의 미니멀한 흰색·남색·파란색 형식을 따르는 5장짜리 EDA 및 Feature Engineering 발표 자료를 만든다.

**산출물:**

- `experiments/gs/eda_feature_engineering_presentation.html`
- `experiments/gs/eda_feature_engineering_presentation.pptx`

## 구성

1. 유전체 변이 데이터의 희소성·불균형·변이 문자열 의미
2. train-only, fold-train-only, NaN 비변이, 하드코딩 금지 원칙
3. 유전자 변이 여부·부담량·기능성 유형·아미노산 치환 방향
4. 유전자×변이유형 및 정확 변이의 Empirical-Bayes 암종별 증거 점수
5. 안전한 변이 표현 → Logistic Regression 흐름과 성능 메시지

## 검증

- HTML을 브라우저에서 열 수 있는 독립 문서로 작성한다.
- PPTX는 편집 가능한 텍스트와 단순 도형으로 생성한다.
- 모든 슬라이드를 PNG로 렌더링해 줄바꿈·겹침·가독성을 확인한다.
- HTML/PPTX 모두 16:9 비율, 템플릿과 동일한 흰 배경·짙은 남색·파란 강조색을 사용한다.
