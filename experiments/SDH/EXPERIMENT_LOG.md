# SDH 실험 통합 로그

이 문서는 SDH가 수행한 전처리 실험을 동일한 흐름으로 정리한 기록이다. 주 지표는
OOF Macro F1이며, 대회 제출 점수(LB)는 별도 항목으로 구분한다.

## 공통 배경과 규칙

- 데이터: train 6,201행, 원본 유전자 열 4,384개
- 기본 검증: Stratified 5-fold
- 초기 탐색: seed 42, 유망 후보: seed 42/52/62
- exp_007 이후 LR 조건: `solver=lbfgs`, `C=0.07`, `max_iter=2000`,
  `class_weight=balanced`
- 각 fold의 train에서만 빈도 목록, hotspot, truncating/recurrent 목록, 상수열 및
  상관 제거 목록을 학습했다. validation과 test에는 `transform`만 적용했다.
- 외부 유전자-암종 annotation은 모델 입력·피처 선택·임계값에 사용하지 않았다.

## 실험 흐름

| 실험 | 질문과 도메인 발상 | 핵심 결과 | 판정 |
| --- | --- | --- | --- |
| exp_001 | 결측·변이 문자열·암종별 분포를 train-only EDA로 확인 | 이후 burden/type/hotspot 설계의 출발점 | 탐색 자료 |
| exp_002 | WT/non-WT 존재 여부만으로 분류 가능한가? | LR 3-seed **0.33738 ± 0.00625**; LGBM **0.28992 ± 0.00065** | baseline |
| exp_003 | 유전자 수·이벤트 수 burden, syntax type, 빈도 필터, hotspot을 추가 | `mutation types` seed42 **0.37803**, hotspot50 **0.37226** | type block 채택 |
| exp_004 | type과 hotspot/min-count 크기를 조합 | type+hotspot50 3-seed **0.37770 ± 0.00704**; LGBM seed42 **0.37904** | hotspot50 우선 |
| exp_005 | exact/codon/gene-type 문자열을 hashing해 희귀 변이를 보존 | hashing 후보는 type+hotspot50을 넘지 못함 | 해싱 보류 |
| exp_006 | 여러 token이 기록된 유전자 수를 burden 3번째 축으로 추가 | burden3 **0.38258 ± 0.00218**, burden2 대비 +0.00488 | 조건부 유망 |
| exp_007 | burden + type + hotspot + truncating + recurrent missense 조합 | functional full **0.43189 ± 0.00325** | 당시 최고 FE |
| exp_008 | functional full의 중복 피처를 상관/중복 제거 | pruning **0.43174 ± 0.00318**, 원본보다 -0.00015 | 성능 목적 채택 안 함 |
| exp_009 | 단백질 표기 구조 A(ref/alt/pair/position)와 행 내부 분포 S를 추가 | A pair **0.46360 ± 0.00109**, 기준 대비 +0.03171 | OOF 1위, LB 재검증 필요 |
| exp_010 | A pair log1p, S, train-only contrast/exact를 순차 누적 | pair log1p **0.48248 ± 0.00098**; 이후 블록은 하락 | pair log1p 채택 |
| exp_011 | B04를 고정하고 독립적인 row-local·class-enrichment FE를 추가 | gene×event-type enrichment **0.52395 ± 0.00202**, Public LB **0.4352596431** | 이전 챔피언 |
| exp_012 | exp011 enrichment의 support·shrinkage·score 구성을 안정화 | **0.52824 ± 0.00187**, Public LB **0.4388787816** | 새 로컬·LB 챔피언 |

## 실험별 해석

### exp_002 — WT/non-WT baseline

행 내부에서 변이가 있는지 없는지만 이진화했다. LR은 다수·소수 암종의 균형을
고려해 `class_weight=balanced`를 사용했고, 이 점수를 이후 모든 개선폭의 기준으로
삼았다. LightGBM은 동일 WT 입력에서 LR보다 낮아, 초기 전처리 비교의 주력 모델은
LR로 유지했다.

### exp_003~005 — 문자열을 구조화하는 단계

EDA에서 관찰한 변이 문자열의 반복 패턴을 유전자 burden, 이벤트 개수, syntax
type으로 바꿨다. 이어서 fold-train에서 반복 token 상위 50개를 hotspot으로 만들었다.
반대로 exact/codon hashing은 희귀 문자열을 보존하지만 희소성과 충돌 때문에
Macro F1이 낮았다. 따라서 문자열 전체를 그대로 보존하기보다, 반복성과 형태를
요약하는 방향을 택했다.

### exp_006~008 — 기능성 블록 조합과 경량화

여러 token이 한 유전자에 몰리는 정도를 burden3으로 추가하고, truncating gene과
hotspot 밖 recurrent missense를 fold-train 목록으로 만들었다. exp_007의
functional full이 가장 높았고, exp_008의 correlation pruning은 피처를 줄였지만
점수를 회복하지 못했다. 즉 단순 중복 제거보다 기능성 신호를 보존하는 편이
유리했다.

### exp_009 — 단백질 표기 구조 확장

외부 단백질 DB를 참조하지 않고, 각 행의 이미 제공된 표기에서 ref/alt 아미노산,
ref→alt pair, 단백질 위치 bin, notation type 분포를 추출했다. 3-seed에서 pair
블록만 추가한 case 06이 가장 안정적이었다. A 전체나 S 전체를 한꺼번에 넣으면
불필요한 열이 섞여 개선폭이 줄었다.

### exp_010 — A pair 표현과 챔피언 요소 누적

exp009의 A pair raw count를 `log1p`로 바꾸자 seed 42에서 0.46286에서 0.48208로
상승했고, seed 42/52/62 평균도 0.48248 ± 0.00098로 안정적이었다. 그 위에 S,
train-only confusion contrast와 train-only exact top-4를 순차 추가하면 각각
점수가 하락했다. 다만 exp010 기준 파이프는 정확한 GS B04가 아니므로 B04 갱신을
확정하지 않고, 후속 실험에서 B04 고정 독립 ablation으로 재검증한다.

### exp_011 — B04 고정 class-enrichment

GS B04 원본 파이프라인과 LR을 그대로 고정하고 burden bin, row profile, gene,
gene×event-type, exact-event class-enrichment를 각각 독립적으로 추가했다. 가장
높았던 피처는 fold-train의 `gene__event_type` log-odds를 26개 암종별 score로
압축한 표현이었다. B04 3-seed 0.47930 ± 0.00253에서 0.52395 ± 0.00202로
0.04465 상승했고, Public LB도 0.38711에서 0.43525로 0.04814 상승했다.

gene enrichment를 함께 넣은 52개 조합은 평균 이득이 +0.00003뿐이고 표준편차가
증가해 채택하지 않았다. 성능 상승은 특히 KIRC/KIPAN과 LGG/GBMLGG 혼동 완화에서
컸다. supervised FE는 outer fold-train 안에서 내부 5-fold OOF cross-fit하여,
학습 행도 자신의 label이 포함된 weight를 직접 받지 않게 했다.

### exp_012 — enrichment 안정화와 하락 클래스 분석

exp011 winner의 최소 support 10, shrinkage 20을 기준으로 support 5/20,
shrinkage 10/50과 LIHC·DLBC·HNSC·LUSC score 제외를 독립 비교했다. seed 42에서
shrinkage 10이 0.52918로 가장 높았고, 3-seed에서도 0.52824 ± 0.00187로
exp011보다 0.00428 높았다. support 5는 0.52667 ± 0.00229로 차선이었다.

support 20과 shrinkage 50은 각각 0.51167, 0.51888로 하락했다. 이미 support 10
필터를 통과한 중빈도 token을 더 강하게 제거하거나 축소하면 유효한 암종 signature도
사라지는 것으로 해석했다. 하락 클래스 score를 개별·동시 제거하는 방식도 해당
클래스 F1을 안정적으로 회복하지 못해 26개 score 전체를 유지했다.

오류는 HNSC→LUSC/STES/CESC, LUSC→LUAD/HNSC/STES, LIHC→LUSC/SARC/STES,
DLBC→STES 등에 집중됐다. 자기 class score는 정답 샘플에서 미탐보다 높았지만,
HNSC와 LUSC는 false positive score 분포가 정답과 많이 겹쳤다. 따라서 단일 score
삭제나 임계값보다 양·음 evidence의 분리 또는 행 내부 상대 score 구조가 다음
후보가 된다.

최신 main에서 exp012 실제 `preprocessing.py`를 직접 불러 permutation-label
sanity check도 수행됐다. 실제 label의 enrichment 이득은 +0.05286, 섞인 label의
이득은 +0.00174로 우연 수준에 머물러 PASS 판정을 받았다.

### exp_012 제출 경로

`experiment.ipynb`의 6번 섹션에 `case_04_shrink10` seed 42 제출 코드를 추가했다.
새 커널에서 해당 섹션만 순서대로 실행할 수 있으며, 전체 train의 enrichment 입력은
내부 5-fold OOF로 만들고 test에는 전체 train에서 학습한 weight를 적용만 한다.

raw train/test는 합치지 않는다. exact-event와 gene×event-type vocabulary는
train에서만 만들고, test가 B04 train 설계행렬에 영향을 주지 않는지 train-only
재구성 행렬과 완전 일치 검사한다. test 결측 개수는 참고용 출력으로만 남기고
고정값 assertion에는 사용하지 않는다. 실행 결과 B04 누수 동치 검사는 PASS,
수렴 경고는 0회였고 B04 8,399개와 enrichment 26개를 합친 총 8,425개 피처로
2,546개 test 예측을 생성했다. 제출 CSV와 metadata JSON은
`experiments/SDH/exp_012_enrichment_stability/results/`에 저장했다. Public LB는
0.4388787816으로 exp011의 0.4352596431보다 0.0036191385 상승했다.

## OOF와 실제 LB의 차이

exp_009 `case_06_plus_A_pair`의 OOF Macro F1은 **0.46360 ± 0.00109**였지만,
전체 train으로 재학습해 제출한 실제 LB는 **0.34238**이었다. 참고한
`biodomain02`의 LB 기록은 **0.35097**로, 현재 제출보다 0.00859 높다.

따라서 OOF 1위 FE를 곧바로 일반화된 최종 해법으로 간주하지 않는다. 가능한 원인은
fold 내부 반복 pair의 분포와 실제 test 분포의 차이, 고차원 pair 열의 과적합,
train/test 표기 형식 차이다. LB 점수는 피처 선택에 사후 사용하지 않고, 다음
실험의 검증 가설을 정하는 참고 기록으로만 사용한다.

반면 exp011은 3-seed CV **0.52395**에서 Public LB **0.4352596431**로 이동해 gap은
약 -0.08870이었지만, B04 대비 개선폭은 CV +0.04465와 LB +0.04814로 거의 그대로
전달됐다. 따라서 gene×event-type enrichment는 exact token보다 test 전이성이 높은
표현으로 판단한다. exp012는 CV에서 추가 +0.00428, Public LB에서 +0.0036191385를
확보했다. CV 개선폭의 약 84.5%가 LB에 전달됐고 CV→LB gap도 약 -0.08936으로
exp011과 거의 같아 shrinkage 10을 새 고정 기준으로 채택한다.

## 현재 결론과 후속 연구

1. 현재 로컬·Public LB 챔피언은 exp012의
   `B04 + gene×event-type enrichment(support10, shrink10)`이다.
2. 3-seed CV는 0.52824 ± 0.00187, Public LB는 0.4388787816이다.
3. exp012 실제 코드 permutation 감사와 seed42 Public LB 확인을 모두 통과했다.
4. 후속 FE는 모델을 바꾸지 않고 enrichment의 양·음 evidence mass, positive share,
   one-vs-rest nonlinear margin과 행 내부 score rank를 독립적으로 검증한다.
5. 모델 파라미터는 LR `C=0.07`, `max_iter=2000`으로 고정하고, 외부 annotation이나
   test 통계를 피처 설계에 사용하지 않는다.
