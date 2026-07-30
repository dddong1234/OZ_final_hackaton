"""scikit-learn 방식의 최종 전처리를 선택할 경우 이 인터페이스로 구현합니다."""


def fit(train_df, target_column, id_column):
    raise NotImplementedError("최종 전처리 파이프라인을 선택해 구현하세요.")


def transform(dataframe, state, target_column, id_column):
    raise NotImplementedError("최종 전처리 파이프라인을 선택해 구현하세요.")
