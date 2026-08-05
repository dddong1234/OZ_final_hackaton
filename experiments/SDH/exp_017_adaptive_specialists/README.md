# SDH exp_017 — adaptive-capacity dynamic specialists

## 실험 질문

exp14의 안전 LR, balanced multiclass LGBM, fold-train 유사 class pair 발견과
LR80/LGBM20을 모두 고정한다. 암종쌍마다 binary specialist의 학습 용량과 적용
조건만 바꾸면 OOF Macro F1이 안정적으로 개선되는가?

## 착안 근거

참고 구조는 서로 다른 혼동쌍에 10-tree specialist와 100-tree specialist를 따로
사용했다. exp14는 동일한 large specialist를 자동 발견 pair 두 개에 적용해 Public
LB 0.4489813603을 기록했지만, pair별 표본 수와 난이도가 다른데도 같은 용량을
사용했다. exp17은 이 비대칭 specialist 용량을 암종명 하드코딩 없이 검증한다.

Focal main model은 exp14에서 balanced multiclass 0.476313보다 낮았으므로 다시
탐색하지 않는다. 모델 파라미터 변화는 binary specialist에만 한정한다.

## 고정 조건

- exp13 safe feature pipeline
- LR: C=0.07, max_iter=2000, class_weight=balanced
- main LGBM: exp14 `main_01_multiclass_balanced`
- pair 발견: outer-fold train의 class별 mutation-gene prevalence cosine similarity
- pair 개수: 상위 2개
- blend: LR 80% + specialist LGBM 20%
- validation/test 통계, 고정 암종명, 외부 annotation 미사용

## Specialist preset

| Preset | trees | learning rate | leaves | min child samples |
| --- | ---: | ---: | ---: | ---: |
| small | 10 | 0.10 | 20 | 20 |
| medium | 40 | 0.05 | 20 | 15 |
| large | 100 | 0.02 | 20 | 10 |

각 outer fold에서 pair마다 세 preset을 한 번씩 학습해 validation 확률을 cache한다.
이후 144개 case는 cache된 확률만 조합하므로 모델을 반복 학습하지 않는다.

## Capacity policy 12종

- 고정 조합 9종: SS, SM, SL, MS, MM, ML, LS, LM, LL
- support-adaptive 3종: pair 표본 수가 100/200/400 이하이면 small, 초과면 large

SS의 첫 글자는 cosine 1위 pair, 두 번째 글자는 2위 pair의 preset이다. exp14
기준은 `fixed_ll`이다.

## Routing 12종

- hard: 메인 예측이 pair에 속하면 specialist 100% 적용
- soft50/soft75: 같은 행에 specialist ratio를 50%/75% 적용
- pair margin: pair 내부 정규화 확률 차이가 0.10/0.20/0.40/0.60/0.80 이하인
  불확실한 행만 hard routing
- global margin: 전체 top1-top2 차이가 0.02/0.05/0.10/0.20 이하인 행만 hard
  routing

총 `12 capacity × 12 routing = 144 cases`다.

## 실행 및 판정

`experiment.ipynb`를 위에서 아래로 실행한다.

1. seed42에서 144개 case 전체 스크리닝
2. exp14 기준 `fixed_ll__hard`와 확률이 중복되지 않는 상위 8개를 자동 선정
3. seed52/62에서 선정 후보만 재학습·확인
4. 3-seed 평균, 표준편차, 최소 delta를 비교

최종 채택 조건은 exp14 `fixed_ll__hard` 대비 3-seed 평균 개선, 최소 seed delta
양수, routed row 수가 0보다 큰 경우다. seed42만 높은 후보는 채택하지 않는다.

결과 CSV는 `results/`에 저장하며 Git에는 커밋하지 않는다.

## 확정 결과

144개 seed42 스크리닝 후 확률이 중복되지 않는 상위 8개와 exp14 기준을
seeds `42/52/62`에서 확인했다.

| case | 3-seed 평균 | exp14 대비 평균 | 최소 seed 증분 | 개선 seed | 판정 |
| --- | ---: | ---: | ---: | ---: | --- |
| `fixed_ls__hard` | **0.540456** | **+0.000611** | **+0.000184** | 3/3 | PASS |
| `fixed_lm__hard` | 0.540221 | +0.000377 | +0.000016 | 3/3 | PASS |
| `fixed_ms__hard` | 0.540141 | +0.000297 | -0.000130 | 2/3 | FAIL |
| `fixed_ll__hard` | 0.539845 | 기준 | 0.000000 | - | baseline |

최종 1위 `fixed_ls__hard`는 cosine 1위 자동 암종쌍에 large specialist, 2위 쌍에
small specialist를 적용하고 main 예측이 해당 pair에 속할 때 hard routing한다.
암종명은 코드에 고정하지 않고 각 outer-fold train에서 pair를 다시 발견한다.

seed별 exp14 대비 결과는 다음과 같다.

| seed | `fixed_ll__hard` | `fixed_ls__hard` | 증분 |
| ---: | ---: | ---: | ---: |
| 42 | 0.543679 | 0.543963 | +0.000284 |
| 52 | 0.540053 | 0.541419 | +0.001367 |
| 62 | 0.535802 | 0.535986 | +0.000184 |

## 판정

- 사전 채택 조건을 통과한 실험상 승자는 `fixed_ls__hard`다.
- pair마다 같은 specialist 용량을 쓰는 것보다 1위 large/2위 small 구성이 세 seed에서
  모두 소폭 높았다.
- 다만 평균 상승이 `+0.000611`로 작아 단독 Public LB 제출은 진행하지 않았다.
- exp14의 안전 구조를 깨지 않는 후속 specialist 후보로만 보관한다.

수렴 및 실행 오류는 없었고, 데이터 누수 계약은 exp14의 fold-train fit-only
전처리와 자동 pair 발견 규칙을 그대로 상속한다.
