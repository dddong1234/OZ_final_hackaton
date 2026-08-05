# H0 Selective-EB 최종 제출 구성

## 채택 근거

동일한 H0 비교에서 고정 `margin=0.05`, 고정 `0.80 / 0.20` 결합을 세 seed로 검증했다.

| Seed | H0 | Selective-EB H0 | Δ |
|---:|---:|---:|---:|
| 42 | 0.544744 | 0.547915 | +0.003171 |
| 777 | 0.538710 | 0.543247 | +0.004537 |
| 2024 | 0.544306 | 0.550605 | +0.006299 |
| 평균 | 0.542587 | 0.547256 | +0.004669 |

12/15 fold가 H0보다 상승했고, 수렴 경고는 0건이었다.

## 전처리와 피처

- **G:** train에서 관찰된 mutation gene binary.
- **B/V/T/R:** 행별 변이량·event-type count·truncating gene·fold-train 반복 missense event.
- **A-pair / S:** 아미노산 치환 방향 log1p count 및 행별 event topology 요약.
- **기존 gene×event-type enrichment:** outer/final train 내부 cross-fit으로 계산한 26차원 암종 점수.
- **Empirical-Bayes enrichment:** 희귀 gene×event-type도 전역 발생률 방향으로 수축해 만든 26차원 암종 증거 점수.
- **Selective gate:** EB LR의 자체 margin이 0.05보다 낮은 행은 non-EB LR 확률을, 그 외는 EB LR 확률을 사용한다.
- **자동 specialist:** full train의 gene mutation centroid 유사도로 암종쌍 2개를 자동 발견하고, LGBM으로 해당 쌍 내부 확률 질량만 재분배한다.

## 규정 계약

- 외부 데이터·외부 annotation·고정 암종/유전자/변이 목록을 쓰지 않는다.
- event vocabulary, recurrent event, EB 가중치, 표준화, specialist 쌍은 **full train만**으로 fit한다.
- test는 위에서 학습한 변환 적용 및 확률 예측에만 사용한다. train/test concat과 test 기반 통계·선택·scaling은 없다.
- WT, 빈 문자열, NaN은 이벤트로 파싱하지 않는다. 제출 audit에서 `nan_as_mutation_count=0`을 확인한다.

## 실행

`exp/exp-h0-selective-eb-submission-01.ipynb`를 위에서 아래로 실행한다. 결과 CSV와 audit JSON은 `experiments/gs/notebooks/submission/`에 생성된다.
