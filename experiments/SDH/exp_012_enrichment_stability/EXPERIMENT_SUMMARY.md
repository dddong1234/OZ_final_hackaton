# exp_012 실험 요약

## 목적

exp011의 `gene×event-type enrichment` 개선을 유지하면서 하락 클래스의 오류를
줄일 수 있는 안전한 표현 설정을 찾는다. 모델 파라미터는 변경하지 않는다.

## 주요 출력

- `leaderboard_seed42.csv`
- `leaderboard_confirmation.csv`
- `target_error_pairs_seed42.csv`
- `target_score_distribution_seed42.csv`
- `target_score_by_true_class_seed42.csv`
- `class_f1_seed42.csv`, `class_f1_confirmation.csv`

## 결과

### seed 42 전체 screen

| 순위 | Case | OOF Macro F1 | exp011 winner 대비 | 판정 |
| ---: | --- | ---: | ---: | --- |
| 1 | shrinkage 10 | **0.52918** | **+0.00279** | 3-seed 확인 |
| 2 | support 5 | 0.52905 | +0.00265 | 3-seed 확인 |
| 3 | HNSC score 제외 | 0.52712 | +0.00072 | 기각 |
| 4 | exp011 winner | 0.52640 | 기준 | 기준 |
| 5 | DLBC score 제외 | 0.52555 | -0.00085 | 기각 |
| 6 | LIHC score 제외 | 0.52528 | -0.00112 | 기각 |
| 7 | LUSC score 제외 | 0.52521 | -0.00119 | 기각 |
| 8 | 하락 4개 score 제외 | 0.52494 | -0.00146 | 기각 |
| 9 | shrinkage 50 | 0.51888 | -0.00752 | 기각 |
| 10 | support 20 | 0.51167 | -0.01473 | 기각 |
| 11 | B04 | 0.47786 | -0.04854 | 기준 |

### 3-seed 확인

| Case | seed 42 | seed 52 | seed 62 | 평균 ± 표준편차 | B04 대비 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B04 | 0.47786 | 0.48286 | 0.47718 | 0.47930 ± 0.00253 | 기준 |
| support 5 | 0.52905 | 0.52737 | 0.52357 | 0.52667 ± 0.00229 | +0.04736 |
| shrinkage 10 | **0.52918** | **0.52991** | **0.52562** | **0.52824 ± 0.00187** | **+0.04893** |

`shrinkage=10`은 exp011 winner의 3-seed `0.52395 ± 0.00202`보다
`+0.00428` 높다. support 5보다도 평균이 `+0.00157` 높고 표준편차가 작다.
모든 실행에서 수렴 경고는 0회였다.

## 해석

1. 기존 shrinkage 20보다 10이 높다는 것은 support 10 이상 gene×type token을
   현재보다 덜 축소할 때 유효한 class signal이 더 잘 보존된다는 뜻이다.
2. support 5도 개선됐지만 shrinkage 10보다 낮고 변동성이 더 크다. 희귀 token을
   더 많이 허용하는 것보다 기존 vocabulary의 가중치를 덜 줄이는 편이 낫다.
3. support 20과 shrinkage 50은 각각 `-0.01473`, `-0.00752`로 크게 하락했다.
   강한 필터링·축소는 유효한 희귀/중빈도 mutation signature까지 제거한다.
4. LIHC, DLBC, HNSC, LUSC score를 하나씩 또는 함께 제외해도 전체 성능이
   개선되지 않았다. 하락의 원인은 해당 score 하나의 존재가 아니라 26개 score와
   B04가 함께 만드는 클래스 경계에 있다.

## 하락 클래스 분석

exp011 winner의 자기 class score는 정답·오답 샘플을 분리하는 방향성은 있었다.

| 클래스 | 정답 score 평균 | 미탐 score 평균 | 정답 확률 평균 | 미탐 확률 평균 |
| --- | ---: | ---: | ---: | ---: |
| LIHC | 0.582 | 0.114 | 0.447 | 0.097 |
| DLBC | 1.272 | 0.463 | 0.488 | 0.080 |
| HNSC | 0.806 | 0.511 | 0.428 | 0.092 |
| LUSC | 1.342 | 0.835 | 0.619 | 0.147 |

다만 HNSC와 LUSC는 false positive score 평균도 각각 `0.646`, `1.275`로 높아
분포가 많이 겹쳤다. DLBC는 전체 support가 38이고 false positive가 5개뿐이라
분산이 크다. 개별 score는 독립 판정값이 아니라 LR이 다른 25개 score 및 B04와
함께 해석해야 하는 상대적 증거다.

주요 오분류는 다음처럼 같은 mutation profile을 공유할 가능성이 큰 클래스에
집중됐다.

- LIHC: LUSC 11, SARC 8, STES 8, LUAD 7, HNSC 6
- DLBC: STES 7, SARC 3, HNSC 3
- HNSC: LUSC 24, STES 18, CESC 18, PAAD 17
- LUSC: LUAD 24, HNSC 11, STES 10

## 최종 판정

최종 채택 후보는
`B04 + gene×event-type enrichment(support=10, shrinkage=10, all 26 scores)`다.

- LR은 기존과 동일하게 `lbfgs`, `C=0.07`, `max_iter=2000`을 유지한다.
- exp012의 `shrinkage=10`은 모델 규제가 아니라 enrichment FE weight 규제다.
- Public LB는 **`0.4388787816`**으로 exp011 `0.4352596431`보다
  **`+0.0036191385`** 높다.
- CV 개선폭 `+0.00428488`의 약 84.5%가 LB에도 전달됐다.
- exp012 실제 코드 permutation-label sanity check도 PASS했다.
- 따라서 exp012를 새 로컬·Public LB 챔피언으로 채택한다.
