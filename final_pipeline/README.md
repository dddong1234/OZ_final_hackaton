# Final Pipeline

팀이 최종 후보를 확정한 뒤 공동 파이프라인을 구성하는 공간입니다.

- scikit-learn 방식이면 `preprocess.py`의 `fit(train_df, target_column, id_column)`과
  `transform(dataframe, state, target_column, id_column)` 인터페이스를 사용할 수 있습니다.
- PyTorch/TensorFlow 방식이면 이 인터페이스를 강제하지 않고 `dataset.py`, `model.py`,
  `train.py`, `inference.py` 등 최종 후보에 맞는 구조를 팀이 합의해 사용합니다.
- 개인 실험 코드를 바로 옮기지 말고 재현 확인과 팀 리뷰를 마친 후보만 반영합니다.
