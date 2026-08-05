# SDH exp_020 — H0 Selective-EB 3-seed 독립 재현

## 한 줄 요약

제공받은 `reproduce_h0_selective_eb_3seed_standalone.py`의 최종 추론 파이프라인에
빠져 있던 **최종 조합 OOF 평가**를 독립 구현해 보고된 3-seed CV `0.564797`을
재현한다. 재현에 성공하면 설정을 전혀 바꾸지 않고 새로운 seeds
`31415/52/62`에서 강건성을 확인한다.

## 왜 이 실험이 우선인가

보고값이 동일한 검증 계약의 Macro F1이라면 기존 H0 `0.543679`보다 약
`+0.0211`, exp19 strict LR+LGBM `0.541299`보다 약 `+0.0235` 높다. XGBoost와
CatBoost의 추가 이득이 `+0.0004` 미만이었던 것과 비교하면 우선순위가 압도적으로
높다.

다만 제공 파일은 최종 test submission은 만들지만 `Selective-EB + specialist`의
OOF 점수를 계산하지 않는다. 따라서 `0.564797`은 그 파일만 실행해서는 검증되지
않는다. exp20은 원본 상수를 동결한 채 이 누락된 평가 계층만 추가한다.

## 최종 결과

### 핵심 결론

- 제공 보고값 `0.564797`은 seed별 F1 평균이 아니라 **세 seed OOF 확률을
  평균한 뒤 계산한 Macro F1**이었다.
- 원본 seeds `42/777/2024`에서 `0.564154`로 재현했다. 보고값과 차이는
  `-0.000643`으로 사전 허용치 `±0.001` 안이다.
- 새로운 seeds `31415/52/62`에서도 확률평균 OOF `0.565292`를 얻었고 세 seed
  모두 같은 seed의 H0보다 상승했다.
- 원본 3-seed 제출의 Public LB는 **`0.48112`**로, 현재 최종 챔피언이다.
- 여섯 seed의 OOF는 `0.566309`로 더 높았지만 Public LB는 `0.4748327994`로
  하락했다. 따라서 6-seed와 후속 9-seed 확장은 기각한다.

### 원본 3-seed 재현

| seed | H0 | 최종 Selective-EB | 증분 |
| ---: | ---: | ---: | ---: |
| 42 | 0.544958 | 0.548042 | +0.003084 |
| 777 | 0.539272 | 0.543252 | +0.003980 |
| 2024 | 0.544277 | 0.550626 | +0.006349 |

- seed별 최종 F1 평균: `0.547307`
- H0 3-seed 확률평균 OOF: `0.557223`
- 최종 3-seed 확률평균 OOF: **`0.564154`**
- H0 대비 확률평균 증분: `+0.006931`
- fold 상승: 15개 중 11개
- Public LB: **`0.48112`**

seed 42의 H0는 제공 기준 `0.543679`보다 `+0.001279` 높아 제공 허용치
`±0.001`을 조금 벗어났다. 다만 exp20의 material guard `±0.005` 안이며, 핵심
최종 3-seed 보고값을 `±0.001` 안에서 재현했으므로 실험을 중단할 차이는 아니라고
판정했다.

### Fresh 3-seed 강건성 확인

| seed | H0 | 최종 Selective-EB | 증분 |
| ---: | ---: | ---: | ---: |
| 31415 | 0.536121 | 0.544970 | +0.008848 |
| 52 | 0.538930 | 0.546221 | +0.007291 |
| 62 | 0.536420 | 0.542608 | +0.006188 |

- seed별 최종 F1 평균: `0.544600`
- H0 3-seed 확률평균 OOF: `0.554970`
- 최종 3-seed 확률평균 OOF: **`0.565292`**
- H0 대비 확률평균 증분: `+0.010322`
- fold 상승: 15개 중 12개

### 6-seed 추가 검증과 기각 근거

| 구성 | OOF Macro F1 | Public LB | 원본 3-seed LB 대비 |
| --- | ---: | ---: | ---: |
| 원본 3-seed `42/777/2024` | 0.564154 | **0.48112** | 기준 |
| 전체 6-seed 동일 가중 평균 | **0.566309** | 0.4748327994 | **-0.0062872006** |

6-seed는 OOF에서 `+0.002156` 상승했지만 test 최종 예측 108개를 바꾸면서
Public LB가 크게 하락했다. 누수 검사와 NaN 검사는 통과했으므로 파일 오류가 아니라
추가 seed가 공개 평가 분포에서 원본 3-seed의 유효한 결정을 희석한 결과로 해석한다.
seed 수가 많을수록 항상 좋아지는 것은 아니며, 9-seed 확장은 진행하지 않는다.

### 안전성 검사

- 제공 원본 SHA-256:
  `969627e00063eebaeac5ba2dd7c04f601ee215cc43da81586d19d7bb18385ccf`
- 모든 outer-fold leakage audit: PASS
- raw train/validation 및 train/test concat: 없음
- test 기반 vocabulary·통계·threshold 선택: 없음
- 고정 암종쌍·exact mutation 규칙: 없음
- NaN을 mutation으로 처리한 행: 0
- Logistic Regression 수렴 경고: 0

## 최종 판정

최종 제출 후보는 **원본 seeds `42/777/2024`의 동일 가중 확률평균**으로 고정한다.
6-seed OOF 상승은 LB로 전이되지 않았으므로 채택하지 않는다. 다음 실험은 seed 추가가
아니라 현 챔피언의 Selective-EB margin과 specialist 결합 비중처럼 소수의
의사결정 경계만 train-only 검증하는 방향을 권장한다.

## 파일

- `provided_pipeline.py`: 사용자가 제공한 standalone 파일의 원문 복사본
- `oof_reproduction.py`: 최종 OOF 평가·집계·감사 구현
- `experiment.ipynb`: 초급자용 단계별 실행 노트북
- `test_oof_reproduction.py`: gate, 80:20, EB cross-fit, 모델 계약 테스트
- `metrics.json`: 최종 재현값·LB·판정 메타데이터
- `results/`: 실행 산출물; Git에서 제외

## 동결된 최종 구조

```text
outer-fold train
  ├─ G+B+V+T+R+A+S + 기존 enrichment
  ├─ non-EB multinomial LR
  ├─ gene×event-type Empirical-Bayes 26-class score
  │    └─ inner 5-fold cross-fit 후 EB LR
  ├─ EB LR margin < 0.05이면 non-EB LR로 복귀
  ├─ balanced multiclass LGBM 400 trees
  └─ train에서 자동 발견한 유사 암종쌍 2개 binary specialist

outer validation
  └─ 0.80 × Selective-EB LR + 0.20 × specialist
```

### 동결 상수

- LR: `C=0.07`, `max_iter=2000`, `class_weight=balanced`
- EB: `alpha=1`, `shrinkage=20`, `clip=4`
- selective margin: `0.05`
- LGBM: 400 trees, lr `0.05`, leaves `25`, min child `10`
- specialist: 100 trees, lr `0.02`, leaves `20`
- final weights: Selective-EB LR `0.80`, specialist `0.20`
- reproduction seeds: `42/777/2024`
- fresh seeds: `31415/52/62`

어떤 파라미터도 결과를 보고 다시 고르지 않는다.

## 누수 방지 계약

각 outer fold에서 다음 항목은 outer train으로만 학습한다.

- exact/gene-type vocabulary
- 활성 유전자와 recurrent missense 선택
- 기존 enrichment weight와 표준화 통계
- EB weight와 표준화 통계
- 자동 specialist 암종쌍
- LR, LGBM, binary specialist 모델

outer validation은 학습된 변환을 적용하고 예측하는 데만 사용한다. exp20 OOF
실행기는 `test.csv`를 읽지 않는다. 암종·유전자·exact mutation을 코드에 고정하지
않는다. selective gate는 한 행의 확률 margin만 사용하므로 행 독립 연산이다.

## 실행 순서

`experiment.ipynb`를 위에서부터 한 셀씩 실행한다.

1. 환경·데이터·제공 원본 SHA 확인
2. seed 42 재현: H0 `0.543679`과의 차이를 먼저 확인
3. seeds 777/2024 재현
4. per-seed 평균과 3-seed 확률평균 OOF를 각각 `0.564797`과 비교
5. 재현 성공 시에만 fresh seeds `31415/52/62` 실행
6. 결과 저장과 판정
7. 6-seed OOF 결합과 제출 gate 확인
8. gate 통과 시에만 6-seed 제출 생성

seed별 결과는 메모리에 남고 실행 직후 `results/`에 checkpoint로 저장한다. 긴 셀
도중 문제가 생겨도 완료된 seed를 다시 돌릴 필요가 없다.

## 1차 재현 판정

- seed42 H0가 제공 허용치 `±0.001`을 벗어나면 audit 경고로 기록
- 단, 차이가 `±0.005`를 넘으면 구현 계약이 달라진 것으로 보고 즉시 중단
- 최종 점수는 두 정의를 모두 기록
  - 세 seed 각각의 Macro F1 평균
  - 세 OOF 확률을 평균한 뒤 계산한 Macro F1
- 둘 중 보고자가 사용한 정의가 `0.564797 ± 0.001`이면 재현 성공
- 15개 fold, 클래스별 F1, 수렴 경고, leakage audit 저장

H0 차이가 작더라도 최종 보고에는 observed/reference/delta를 함께 남긴다.

## 2차 강건성 판정

재현 후 fresh seeds에서는 다음을 요구한다.

- 세 seed 모두 최종 조합이 같은 seed의 H0보다 상승
- 평균 증분이 양수
- 15개 fold 중 과반에서 상승
- 수렴 경고 및 leakage audit 이상 없음

fresh seed 결과가 약해도 reproduction 설정을 수정하지 않는다. 그 경우 보고값은
선택 seeds에 특화된 결과로 기록하고 최종 파이프라인 채택을 보류한다.
