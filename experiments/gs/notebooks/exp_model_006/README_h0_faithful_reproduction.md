# H0 faithful reproduction audit

실행 노트북은 `exp/exp-h0-faithful-reproduction-01.ipynb`이고 실행기는 `common/run_h0_faithful_reproduction.py`다.

이 실행기는 다른 실험 폴더를 import하지 않는다. train-only로 vocabulary, recurrent missense, enrichment, 표준화, 자동 specialist pair를 fit한다. seed42 OOF 기준이 `0.543679 ± 0.001`이면 `reproduced`, 그렇지 않으면 `baseline_not_reproduced`와 `block_downstream_experiments=true`를 저장한다.

전체 CV 전에는 노트북의 smoke 셀을 실행한다. 전체 결과는 `result/exp-h0-faithful-reproduction-01_seed42_*`에 저장된다.
