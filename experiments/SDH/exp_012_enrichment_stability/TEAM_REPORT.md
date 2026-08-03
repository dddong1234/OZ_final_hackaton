# SDH exp_012 팀 공유 보고서 — class-enrichment 안정화

## 요약

- exp011 챔피언인 `B04 + gene×event-type class-enrichment 26개`를 고정하고,
  enrichment의 최소 support·shrinkage와 class score 구성만 독립적으로 바꿨다.
- 최종 후보는 **support 10, shrinkage 10, class score 26개 전체 유지**다.
- 3-seed OOF Macro F1은 B04 `0.47930 ± 0.00253`, exp011 winner
  `0.52395 ± 0.00202`, exp012 후보 **`0.52824 ± 0.00187`**다.
- exp012 후보는 B04 대비 **+0.04893**, exp011 대비 **+0.00428** 개선됐다.
- support 5도 `0.52667 ± 0.00229`로 유망했지만 shrinkage 10보다 평균이 낮고
  변동성이 더 컸다.
- LIHC, DLBC, HNSC, LUSC score를 빼는 방식은 해당 클래스 F1을 안정적으로
  회복시키지 못했고 전체 Macro F1도 하락했다. 따라서 26개 score를 유지한다.
- 모든 실행에서 LR 수렴 경고는 0회였다.
- exp012 후보는 아직 Public LB에 제출하지 않았다. 제출 검증이 끝난 현재
  챔피언은 exp011의 Public LB `0.43525`다.

## 1. 실험 배경

exp011에서는 B04 위에 fold-train에서 학습한 gene×event-type class-enrichment
score 26개를 추가해 큰 개선을 얻었다. 하지만 클래스별 분석에서 LIHC, DLBC,
HNSC, LUSC는 B04보다 F1이 하락했다.

exp012는 모델이나 B04를 바꾸지 않고 다음 세 질문만 검증했다.

1. enrichment token을 지나치게 버리거나 축소하고 있지는 않은가?
2. 하락 클래스의 자기 score가 오히려 잘못된 경계를 만들고 있는가?
3. 네 클래스의 score 분포와 실제 오분류 상대는 어떤 구조를 보이는가?

각 case는 exp011 winner에서 독립적으로 한 요소만 바꿨다. 따라서 결과 차이를
support, shrinkage 또는 score 제외 효과로 직접 해석할 수 있다.

## 2. 고정한 파이프라인

### 2.1 B04

B04의 gene presence, burden 3종, event type count, truncation, recurrent missense,
아미노산 치환 방향, 행 내부 구조, confusion-pair contrast를 그대로 사용했다.

### 2.2 모델과 검증

- Logistic Regression `solver="lbfgs"`
- `C=0.07`, `max_iter=2000`, `class_weight="balanced"`
- Stratified 5-fold
- 전체 screen: seed 42
- 최종 확인: seed 42, 52, 62
- 평가 지표: OOF Macro F1

### 2.3 enrichment 계산

token은 `gene__event_type`이다. 예를 들면 `TP53__MISSENSE`,
`APC__NONSENSE`다. class `c`와 token `t`의 fold-train log-odds 차이를 구한 뒤
support 기반 shrinkage를 적용한다.

```text
raw_weight(c,t)
  = log((n_pos + 1) / (N_pos - n_pos + 1))
  - log((n_neg + 1) / (N_neg - n_neg + 1))

weight(c,t)
  = clip(raw_weight(c,t) × support/(support + shrinkage), -4, 4)

score_c(x)
  = Σ active_token weight(c,t) / sqrt(number of active tokens)
```

`min_support`는 weight 학습에 남길 token의 fold-train 최소 출현 샘플 수다.
`shrinkage`는 낮은 support token의 weight를 0 쪽으로 줄이는 FE 규제다.
LR 규제 `C=0.07`과는 다른 값이다.

## 3. 누수 방지

enrichment는 label을 쓰므로 outer fold마다 내부 5-fold cross-fit을 수행했다.

1. outer train을 inner train/holdout으로 나눈다.
2. inner train label로 token weight를 학습한다.
3. inner holdout에는 학습된 weight를 적용만 한다.
4. 이를 반복해 outer train용 OOF enrichment를 완성한다.
5. outer train 전체로 weight를 다시 학습해 outer validation에 적용한다.
6. 표준화 평균·표준편차도 outer train OOF score에서만 계산한다.

따라서 validation label과 통계는 피처 생성·선택·표준화에 사용하지 않았다.

## 4. 실험 구성

| Case | support | shrinkage | 유지 score | 목적 |
| --- | ---: | ---: | --- | --- |
| 00 | - | - | - | B04 기준 |
| 01 | 10 | 20 | 26개 | exp011 winner 재현 |
| 02 | 5 | 20 | 26개 | 희귀 token 추가 허용 |
| 03 | 20 | 20 | 26개 | token 필터 강화 |
| 04 | 10 | 10 | 26개 | weight 축소 완화 |
| 05 | 10 | 50 | 26개 | weight 축소 강화 |
| 06~09 | 10 | 20 | 각 25개 | LIHC/DLBC/HNSC/LUSC score 개별 제외 |
| 10 | 10 | 20 | 22개 | 하락 4개 score 동시 제외 |

## 5. seed 42 screen 결과

| 순위 | Case | OOF Macro F1 | Accuracy | exp011 winner 대비 |
| ---: | --- | ---: | ---: | ---: |
| 1 | shrinkage 10 | **0.52918** | **0.53282** | **+0.00279** |
| 2 | support 5 | 0.52905 | 0.53266 | +0.00265 |
| 3 | HNSC score 제외 | 0.52712 | 0.53056 | +0.00072 |
| 4 | exp011 winner | 0.52640 | 0.52879 | 기준 |
| 5 | DLBC score 제외 | 0.52555 | 0.52798 | -0.00085 |
| 6 | LIHC score 제외 | 0.52528 | 0.52766 | -0.00112 |
| 7 | LUSC score 제외 | 0.52521 | 0.52798 | -0.00119 |
| 8 | 하락 4개 score 제외 | 0.52494 | 0.52766 | -0.00146 |
| 9 | shrinkage 50 | 0.51888 | 0.51943 | -0.00752 |
| 10 | support 20 | 0.51167 | 0.50895 | -0.01473 |
| 11 | B04 | 0.47786 | 0.45847 | -0.04854 |

## 6. 3-seed 확인

| Case | seed 42 | seed 52 | seed 62 | 평균 ± 표준편차 | B04 대비 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B04 | 0.47786 | 0.48286 | 0.47718 | 0.47930 ± 0.00253 | 기준 |
| support 5 | 0.52905 | 0.52737 | 0.52357 | 0.52667 ± 0.00229 | +0.04736 |
| shrinkage 10 | **0.52918** | **0.52991** | **0.52562** | **0.52824 ± 0.00187** | **+0.04893** |

shrinkage 10은 모든 seed에서 support 5와 비슷하거나 높고, 평균은 `+0.00157`,
표준편차는 `-0.00042`다. exp011 winner 대비 평균 개선은 `+0.00428`이다.

## 7. support와 shrinkage 해석

### 7.1 support

- support 5: exp011보다 `+0.00265`로 희귀 token 일부가 실제 신호를 제공했다.
- support 20: exp011보다 `-0.01473`으로 큰 폭 하락했다.

즉 support 10 근처의 중빈도 gene×type token이 중요한데, 이를 20으로 올리면
희귀 클래스 signature가 과도하게 사라진다. support 5의 추가 token도 도움이
되지만 seed 변동성까지 고려하면 기존 support 10을 유지하는 편이 낫다.

### 7.2 shrinkage

- shrinkage 10: exp011보다 `+0.00279`
- shrinkage 50: exp011보다 `-0.00752`

기존 shrinkage 20은 다소 강했다. support 10 이상으로 이미 한 번 걸러진 token의
가중치를 조금 덜 줄이면 class signature가 보존된다. 반대로 50까지 높이면 서로
다른 클래스의 점수가 평평해져 구분력이 감소한다.

## 8. class score 제외 결과

자기 class score가 하락 원인이라면 그 score를 제거했을 때 해당 클래스 F1이
회복돼야 한다. 실제 결과는 그렇지 않았다.

| 제외 score | exp011 해당 클래스 F1 | 제외 후 F1 | 변화 | 전체 Macro F1 변화 |
| --- | ---: | ---: | ---: | ---: |
| LIHC | 0.41398 | 0.40000 | -0.01398 | -0.00112 |
| DLBC | 0.54237 | 0.54237 | 0.00000 | -0.00085 |
| HNSC | 0.37642 | 0.37037 | -0.00605 | +0.00072 |
| LUSC | 0.54762 | 0.54286 | -0.00476 | -0.00119 |

HNSC score 제외의 전체 Macro F1은 소폭 상승했지만 HNSC 자체 F1은 오히려
하락했다. 이는 다른 클래스의 경계가 우연히 조금 개선된 결과다.

네 score를 동시에 제외했을 때 HNSC는 `+0.00409`, LUSC는 `+0.00608`이지만,
LIHC는 `-0.01398`, DLBC는 변화가 없고 전체 Macro F1은 `-0.00146`이다.
따라서 하락 클래스 score를 제거하는 전략은 채택하지 않는다.

## 9. score 분포 분석

exp011 winner seed 42 OOF에서 각 클래스의 자기 enrichment score와 최종 LR
확률을 정답/미탐/false positive로 나눴다.

| 클래스 | 구간 | n | score 평균 | 해당 클래스 확률 평균 |
| --- | --- | ---: | ---: | ---: |
| LIHC | 정답 | 77 | 0.582 | 0.447 |
|  | 미탐 | 81 | 0.114 | 0.097 |
|  | false positive | 137 | 0.250 | 0.233 |
| DLBC | 정답 | 16 | 1.272 | 0.488 |
|  | 미탐 | 22 | 0.463 | 0.080 |
|  | false positive | 5 | 1.242 | 0.527 |
| HNSC | 정답 | 83 | 0.806 | 0.428 |
|  | 미탐 | 140 | 0.511 | 0.092 |
|  | false positive | 135 | 0.646 | 0.292 |
| LUSC | 정답 | 115 | 1.342 | 0.619 |
|  | 미탐 | 63 | 0.835 | 0.147 |
|  | false positive | 127 | 1.275 | 0.488 |

네 클래스 모두 정답 샘플의 평균 score가 미탐보다 높다. 즉 자기 score의 방향은
맞다. 다만 HNSC와 LUSC는 false positive 분포가 정답에 가깝고, DLBC는 support
38로 표본이 매우 작다. 단일 score 임계값으로 고치기 어려운 이유다.

또한 다른 실제 클래스에서도 특정 class score가 높게 나왔다. 예를 들어 LUSC
score 평균은 실제 LUSC에서 1.163이지만 SKCM에서는 1.524, STES에서는 0.897,
LUAD에서는 0.803이었다. 이는 enrichment score가 정답 확률이나 target encoding
그 자체가 아니라 mutation signature의 한 좌표임을 보여준다. LR은 26개 좌표의
상대 패턴과 B04 피처를 함께 이용한다.

## 10. 주요 오분류 상대

| 실제 클래스 | B04 주요 오분류 | exp011 enrichment 주요 오분류 |
| --- | --- | --- |
| LIHC | TGCT 12, PRAD 8, KIPAN 8, OV 8 | LUSC 11, SARC 8, STES 8, LUAD 7 |
| DLBC | PCPG 3, KIPAN 2, TGCT 2 | STES 7, SARC 3, HNSC 3 |
| HNSC | CESC 21, PAAD 20, OV 17, STES 14 | LUSC 24, STES 18, CESC 18, PAAD 17 |
| LUSC | LUAD 24, HNSC 14, STES 12 | LUAD 24, HNSC 11, STES 10 |

enrichment가 KIRC/KIPAN, LGG/GBMLGG 같은 큰 혼동을 해결한 뒤 남은 오류가
HNSC/LUSC/LUAD/STES처럼 서로 일부 mutation profile을 공유하는 그룹으로
집중된 것으로 해석한다. DLBC는 38개뿐이어서 STES 7건만으로 F1 변동이 크다.

## 11. shrinkage 10의 클래스별 변화

seed 42에서 exp011 winner 대비 가장 크게 오른 클래스는 다음과 같다.

| 클래스 | exp011 F1 | shrinkage 10 F1 | 변화 |
| --- | ---: | ---: | ---: |
| GBMLGG | 0.51560 | 0.54040 | +0.02479 |
| KIRC | 0.65396 | 0.67155 | +0.01760 |
| LGG | 0.63853 | 0.65038 | +0.01184 |
| LAML | 0.51250 | 0.52063 | +0.00813 |
| KIPAN | 0.59035 | 0.59729 | +0.00694 |
| STES | 0.55462 | 0.55989 | +0.00527 |

LIHC는 `-0.00647`, HNSC는 `-0.00085`, DLBC와 LUSC는 변화가 없다. 즉 새
shrinkage의 Macro F1 개선은 네 하락 클래스를 직접 복구해서가 아니라 기존에
강했던 confusion-pair 및 여러 중간 클래스의 경계를 더 개선해서 발생했다.

## 12. 최종 판정

### 채택

`B04 + gene×event-type class-enrichment`

- `min_support=10`
- `shrinkage=10`
- 26개 class score 전체 유지
- LR `lbfgs`, `C=0.07`, `max_iter=2000` 유지

### 기각

- support 20: 유효 token을 과도하게 제거
- shrinkage 50: class signature를 과도하게 축소
- 하락 클래스 score 제외: 해당 클래스 회복이 없고 전체 성능도 하락

### 다음 단계

1. shrinkage 10 구현에 permutation-label sanity check 적용
2. 동일 seed 42 제출 파일을 만들어 Public LB 확인
3. LB가 유지되면 exp011 winner 대신 exp012 설정을 새 고정 기준으로 채택
4. LIHC/HNSC/LUSC 보완은 score 제거가 아니라 fold-train 내부의 pairwise margin
   또는 독립적인 row-local 피처로 검증

현재 단계에서는 exp012가 **새 로컬 CV 후보**이고, Public LB까지 확인된 최종
챔피언은 exp011이다.
