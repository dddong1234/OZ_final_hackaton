# 3-seed Selective-EB + 자동 Specialist 제출 정리

## 결과

- **현재 팀 기준 Public LB: 0.48112**
- 제출 방식: seed `42 / 777 / 2024` full-train 확률을 동일하게 평균한 bagging
- Selective-EB branch 안정성: 3-seed OOF에서 H0 대비 평균 `+0.004669`, 상승 fold `12/15`

## 무엇을 했나

단백질 변이 문자열을 단순 WT/변이 binary로만 쓰지 않고, 변이의 양·유형·구조·아미노산 치환 방향·암종별 evidence를 함께 피처로 만들었다.

1. 유전자별 mutation 여부
2. 전체 변이 수, 기능성 변이 유형별 count, truncating 변이 수
3. fold-train에서 반복된 missense exact event
4. 아미노산 치환 방향(ref→alt) 380종의 `log1p` count
5. 한 유전자 내 복수 event, event-type 다양성, entropy 등 topology 8개
6. `gene×event-type`이 각 암종을 지지하는 정도를 26개 evidence score로 압축
7. 희귀 event를 과신하지 않도록 Empirical-Bayes 수축 score 추가

## 모델 결합

- 구조화 피처 + 일반 enrichment LR
- 구조화 피처 + Empirical-Bayes enrichment LR
- EB가 저마진(Top-1−Top-2 < 0.05)일 때는 일반 LR로 대체
- 자동으로 발견한 유사 암종쌍 2개를 LGBM specialist가 쌍 내부에서만 재분류
- 최종 확률: `0.80 × selective-EB LR + 0.20 × specialist LGBM`
- 세 seed의 final probability를 `1/3`씩 평균해 제출

## 왜 의미가 있나

같은 유전자 변이라도 missense인지 truncating인지에 따라 암종 신호가 다를 수 있다. 이 파이프라인은 변이 event를 암종별 evidence로 압축하면서, 희귀 event는 전체 발생률 쪽으로 수축해 과적합 위험을 줄인다.

## 규정·누수 방지

- 외부 데이터·외부 annotation 사용 없음
- 고정 암종명·유전자명·exact mutation 목록을 규칙으로 사용하지 않음
- vocabulary, 반복 event, evidence 통계, standardization, specialist 쌍은 train에서만 학습
- test는 학습된 변환을 적용해 예측만 수행
- train/test concat 없음
- WT·빈 문자열·NaN은 mutation event가 아님
- audit에서 `leakage_check=True`, `nan_as_mutation_count=0` 확인

## 재현

`reproduce_h0_selective_eb_3seed_standalone.py` 하나만 실행하면 submission CSV와 audit JSON이 생성된다.
