# exp-gs-002-14 — 고정 3모델 앙상블 계획

## 목표

새 피처·파라미터·가중치 탐색 없이, 이미 검증된 세 확률을 고정 비율로 합쳤을 때 13번 최고 앙상블을 넘는지 단 한 번 검증한다.

## 고정 구성

- 08 primary LR: `0.50`
- event-token TF-IDF multinomial LR: `0.25`
- event-token TF-IDF OVR LR: `0.25`
- candidate, `min_df=3`, `sublinear_tf=True`, `C=0.07`, `max_iter=2000`, 5-fold, seeds `42/2024/777`은 13번과 동일하다.
- OOF에서는 train.csv만 읽고 fold-train만으로 feature matrix·TF-IDF vocabulary·IDF·모델을 fit한다.

## 비교와 판정

- 비교 기준: 13번 primary+OVR 0.5/0.5 앙상블 `0.483130 ± 0.002638`.
- 저장: 3-way Macro F1/Accuracy, 13 대비 seed별 delta, 클래스별 F1, feature/vocabulary 수, 경고, leakage/NaN audit.
- 채택: 3-seed 평균이 기준을 넘고 seed별 큰 하락이 없으며 모든 안전성 검증이 통과할 때.
- 미검출: 그 외. 이 결과로 가중치·min_df·피처·모델 파라미터를 다시 탐색하지 않는다.

## 구현 순서

1. 확률 행 합을 보존하는 고정 0.50/0.25/0.25 blend 단위 테스트를 작성하고 실패를 확인한다.
2. 공통 runner에 순수 blend helper와 3-way OOF 결과 필드를 추가한다.
3. `--event-tfidf-three-way` CLI와 `exp-gs-002-14.ipynb`를 추가한다.
4. 문법·단위 테스트·NaN parser self-check·노트북 JSON을 검증한다. 실제 OOF는 사용자가 실행한다.
