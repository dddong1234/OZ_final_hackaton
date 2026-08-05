# H0 faithful reproduction audit

이 실행기는 팀 디렉토리를 import하지 않고 GS 내부 코드만으로 exp013/014의 안전 H0를 재현한다.

- exp013: mutation binary, burden 3개, event-type count, truncation, fold-train recurrent missense exact event, A-pair log1p, S topology 8개, support 10 이상 gene×event-type enrichment, inner 5-fold cross-fit 및 train-OOF 표준화.
- exp014: balanced multiclass LGBM, outer-fold train mutation profile로 자동 발견한 유사 암종쌍 2개, predicted-only hard specialist, pair probability mass 보존.
- 최종: LR 0.80 + hard-specialist LGBM 0.20.

seed42 기준값은 LR `0.526130`, LGBM hard specialist `0.492332`, blend `0.543679`이다. 세 점수 모두 ±0.001 이내이고 누수·NaN·수렴 계약을 통과할 때만 `reproduction_pass=true`가 된다. 실패 시 H1/H2-S 점프 실험을 실행하지 않는다.
