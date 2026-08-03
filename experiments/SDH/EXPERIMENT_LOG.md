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

## OOF와 실제 LB의 차이

exp_009 `case_06_plus_A_pair`의 OOF Macro F1은 **0.46360 ± 0.00109**였지만,
전체 train으로 재학습해 제출한 실제 LB는 **0.34238**이었다. 참고한
`biodomain02`의 LB 기록은 **0.35097**로, 현재 제출보다 0.00859 높다.

따라서 OOF 1위 FE를 곧바로 일반화된 최종 해법으로 간주하지 않는다. 가능한 원인은
fold 내부 반복 pair의 분포와 실제 test 분포의 차이, 고차원 pair 열의 과적합,
train/test 표기 형식 차이다. LB 점수는 피처 선택에 사후 사용하지 않고, 다음
실험의 검증 가설을 정하는 참고 기록으로만 사용한다.

## 현재 결론과 후속 연구

1. 로컬 검증 최고 FE는 `functional full + A pair`다.
2. 실제 제출에서는 OOF-LB gap이 커서 pair 블록 단독 채택을 확정하지 않는다.
3. 다음에는 pair vocabulary를 train-fold 빈도 기준으로 축소하는 ablation, pair
   count의 이진화/로그 변환, fold별 pair 안정성(등장률·분산) 검사를 우선한다.
4. 그 전까지 모델 파라미터는 LR `C=0.07`, `max_iter=2000`으로 고정하고, 새
   외부 annotation이나 test 통계를 피처 설계에 사용하지 않는다.
