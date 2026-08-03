# exp-gs-002-04: H-AS exact mutation 4개 ablation

## 목적

현재 최고 후보 `H-AS-LR-exact`의 Macro F1 개선을 만든 exact mutation 4개의 개별 기여를 확인한다. 성능에 기여하지 않거나 seed 간 방향이 불안정한 변이는 최종 제출 후보에서 제외할 근거를 만든다.

## 비교 후보

- 기준: `H-AS-LR-exact` (BRAF V600E, IDH1 R132H, PIK3CA H1047R, PIK3CA E545K)
- `H-AS-LR-exact-minus-BRAF-V600E`
- `H-AS-LR-exact-minus-IDH1-R132H`
- `H-AS-LR-exact-minus-PIK3CA-H1047R`
- `H-AS-LR-exact-minus-PIK3CA-E545K`

## 고정 조건

- LogisticRegression: `lbfgs`, `C=0.07`, `max_iter=2000`, `class_weight=balanced`
- StratifiedKFold 5-fold, seeds `42`, `2024`, `777`
- H-AS backbone: `G+B+V+T+R+A+S`
- fold-train only로 모든 recurrent/feature-selection 통계 산출
- test NaN은 mutation으로 세지 않음: `nan_as_mutation_count=0`
- test는 fit/통계/피처선택에 사용하지 않음: `leakage_check=True`

## 결과 및 판정

후보별 seed OOF Macro F1, 3-seed mean/std, 기준 대비 delta, 수렴 경고 수, feature count를 저장한다. 제거 후보가 기준보다 3-seed 평균에서 개선되고 seed별 방향도 일관될 때만 해당 변이 제거 후보를 다음 제출 후보로 채택한다. 그 외에는 exact 4개를 유지한다.

## 범위 제외

새 hotspot 탐색, 혼동쌍 피처, 다중 블록 조합은 이 실험에서 수행하지 않는다.
