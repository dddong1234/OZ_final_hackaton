# H0 내부 구성요소 보완성 감사

## 목적

현재 규정 안전 H0를 바꾸지 않고, 이미 같은 outer-fold train에서 생성되는 확률 가지가 서로 다른 오류를 보완하는지 확인한다. 이 실험은 점프 모델 탐색이 아니라 마지막 소폭 안정화 감사다.

## 고정 비교 후보

| 후보 | 구성 |
|---|---|
| H0_selective_EB | Selective-EB LR 80% + automatic LGBM specialist 20% |
| H0_non_EB | non-EB LR 80% + 동일 specialist 20% |
| H0_EB | EB LR 80% + 동일 specialist 20% |
| equal_H0_non_EB | H0_selective_EB와 H0_non_EB의 고정 0.5/0.5 평균 |
| equal_H0_EB | H0_selective_EB와 H0_EB의 고정 0.5/0.5 평균 |

가중치·threshold·피처는 탐색하지 않는다. specialist는 각 outer-fold train 안에서 기존 H0와 동일하게 자동 탐지·학습된다.

## 규정 및 누수 계약

- seed42 OOF에서는 `train.csv`만 읽고 test는 읽지 않는다.
- vocabulary, EB 통계, specialist, 스케일링은 모두 outer-fold train에서만 fit한다.
- validation은 transform 및 평가에만 사용한다.
- NaN·WT·공백은 event가 아니다.
- 고정 암종명·유전자명·exact mutation 목록을 쓰지 않는다.

## 판정

최고 fixed blend가 다음을 모두 만족할 때만 별도 3-seed 검증 후보가 된다.

- H0 대비 Macro F1 `+0.003` 이상
- 5개 fold 중 4개 이상 상승
- H0 오답 회복 수 > H0 정답 훼손 수
- H0 저마진 행 Macro F1 하락이 `-0.003` 이내

이 조건을 충족하지 않으면 H0를 그대로 유지하고, 이 앙상블 축은 종료한다. 제출 파일은 이 감사에서 만들지 않는다.
