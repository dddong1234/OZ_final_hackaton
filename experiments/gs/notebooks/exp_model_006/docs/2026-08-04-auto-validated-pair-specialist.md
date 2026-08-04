# exp-auto-validated-pair-specialist-01

H0의 확률을 기준으로, outer-train inner OOF에서 실제 양방향 혼동이 많은 암종쌍만 후보로 찾는다. 후보별 binary specialist는 inner validation에서 pair F1 상승과 `recovered > broken`을 모두 만족해야 채택된다.

선택 쌍은 서로 겹치지 않는 최대 두 개다. outer validation에서는 H0 Top-2가 선택 쌍과 정확히 일치한 행에만 specialist를 적용하며, 해당 쌍의 기존 확률 질량은 보존한다. 모든 탐지·선택·전처리·학습은 outer-fold train 안에서만 수행하고 test는 읽지 않는다.
