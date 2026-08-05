# exp_model_006 — H2-S 공유형 evidence-shape pairwise ranking

`exp-h2-evidence-shape-01`은 안전 H0의 확률을 버리지 않고, 환자×암종 후보별 EB 증거 형태 19개를 이용해 26개 전체 후보의 순위를 잔차 보정한다.

- 실행 노트북: `exp/exp-h2-evidence-shape-01.ipynb`
- 실행기: `common/run_h2_evidence_shape_pairwise.py`
- 전체 CV 전 스모크: `python common/run_h2_evidence_shape_pairwise.py --smoke`
- 결과: `result/`

seed42 screen은 train만 읽는다. 모든 vocabulary·EB·자동 specialist·pairwise ranker는 outer-fold train 내부에서만 fit하며, test는 어떤 단계에서도 읽거나 결합하지 않는다.

H0 OOF가 안전 기준 `0.543679 ± 0.001`을 재현하지 못하면 실행기는 `baseline_not_reproduced`로 판정한다. 이 경우 H2-S delta를 채택 근거로 쓰지 않고 H0 재현 차이부터 확인한다.
