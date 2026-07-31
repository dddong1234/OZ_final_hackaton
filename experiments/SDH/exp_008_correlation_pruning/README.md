# SDH exp_008 — 기능성 피처 상관 제거

exp_007 최고 FE의 truncating/recurrent 피처 중 중복도가 높은 열을 fold train에서만
제거한다. LR은 `C=0.07`, `max_iter=2000`으로 고정한다.

| Case | 처리 |
| --- | --- |
| 01 | exp_007 functional full 기준 |
| 02 | 기능성 피처의 완전 중복 제거 |
| 03 | 기능성 피처끼리 abs(corr) >= 0.99 제거 |
| 04 | 기능성 피처끼리 abs(corr) >= 0.95 제거 |
| 05 | 기능성 피처끼리 abs(corr) >= 0.90 제거 |
| 06 | 0.95 + 같은 gene/hotspot 기본 피처와 중복 제거 |

상관계수와 제거 목록은 매 fold의 학습 분할에서만 계산한다. 노트북에서 각 case를
한 셀씩 실행하고 상위 후보만 3-seed로 확인한다.
