# Fold-aligned 3-seed bagging audit

기존 `0.564797`은 서로 다른 CV split의 OOF를 행 단위로 평균한 값이라 유효한 OOF 성능이 아니다. 이 감사는 outer fold를 seed 42로 고정하고, 각 fold-train에서 model seed `42/777/2024`를 각각 학습한다. 같은 validation 행의 세 확률만 `1/3`씩 평균한다.

- 기준: H0 Selective-EB (`0.80 × LR branch + 0.20 × automatic LGBM specialist`)
- train만 읽음; test 미열람·미결합
- vocabulary, Empirical-Bayes, 표준화, specialist는 outer fold train만으로 fit
- WT/빈 문자열/NaN은 event가 아니며 `nan_as_mutation_count=0`
- 새 가중치나 threshold 탐색 없음

따라서 bagged OOF의 각 행은 세 모델 모두에서 validation 행이다. 결과는 제출용 full-train 3-seed bagging을 직접 보장하지는 않지만, 그 bagging의 train-only 안정성을 검증한다. H1은 이 감사 결과를 기준으로만 진행한다.
