# H0 + Train-only Prototype Similarity Screen

## 가설

현재 H0 Selective-EB는 각 변이 token이 각 암종을 지지하는 정도를 합산한다. 이 실험은 동일한 변이 구성을 가진 환자가 같은 암종에 속하는 국소적 패턴을 별도 신호로 사용한다.

원문/canonical profile 감사에서 동일 profile의 가중 purity가 약 0.918로 높았다. 따라서 class prototype 유사도는 H0와 다른 오류를 낼 가능성이 있다.

## 입력과 처리

각 outer-fold train에서 자동으로 다음을 생성한다.

- mutation 문자열을 gene×functional-type 및 gene×exact-event token으로 파싱
- token 문서 빈도로 IDF 계산
- 환자별 L2-normalized token profile
- 26개 암종별 평균 profile prototype 및 train class prior

validation에는 고정된 fold-train vocabulary/IDF/prototype을 적용해 26개 cosine similarity 기반 확률을 계산한다. apply-only token은 vocabulary를 확장하지 않고 무시한다.

## 비교 구성

- H0_selective_EB: 기존 frozen H0
- prototype_similarity: prototype 확률 단독
- H0_plus_prototype: 고정 0.80 × H0 + 0.20 × prototype

0.80/0.20은 실행 전에 고정한다. 비율, token 목록, threshold 탐색은 하지 않는다.

## 안전 계약

- seed42 OOF에서는 train만 읽는다.
- train/test 결합, test 기반 vocabulary/통계/스케일링은 없다.
- 모든 prototype 통계와 H0 supervised feature는 outer-fold train에서만 fit한다.
- WT·빈 문자열·NaN은 event가 아니며 NaN mutation count는 0이어야 한다.
- 고정 암종명·유전자명·mutation 목록은 사용하지 않는다.
- 실행 중 fold별 checkpoint를 기록하고 큰 sparse matrix는 fold 종료 때 해제한다.

## 승격 조건

seed42 screen에서 아래를 모두 충족해야 3-seed 검증 후보로 올린다.

1. H0 대비 OOF Macro F1 +0.015 이상
2. 5개 fold 중 4개 이상 상승
3. H0 low-margin 구간 Macro F1 하락이 -0.003 이내
4. H0 오답 회복 수가 기존 정답 손상 수보다 큼

그 외 결과는 미검출 또는 기각으로 기록하며 제출 파일은 만들지 않는다.

