# Frozen Biomedical Event Encoder Plan

**Goal:** Compare a fixed PubMedBERT event-string representation with the fixed P1+EB LR baseline without using test-derived statistics or external biological annotation.

**Fixed contract:** `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`; frozen weights; template `gene {gene} type {type} ref {ref} position {pos} alt {alt}`; mean/max pooling; fold-train StandardScaler+LR; fixed blend `.75 P1+EB + .25 encoder`.

**Files:**
- `common/frozen_event_encoder.py`: event sentence construction and pooling.
- `common/run_frozen_encoder.py`: seed-42 E0/E1/E2 OOF runner with dependency gate.
- `common/test_frozen_event_encoder.py`: sentence/pooling unit tests.
- `exp/exp-frozen-biomedical-encoder-01.ipynb`: dependency gate and experiment runner.

**Safety:** no test read during candidate selection; no tokenizer/vocabulary fit; no fine-tuning; no external annotation/sequence; WT/NaN create zero events; final test inference is not implemented in this screening runner.

**Promotion:** seed 42 E2 delta >= .010 and >=4/5 fold gains before 3-seed confirmation.
