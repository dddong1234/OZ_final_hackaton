"""
트랙 A 전처리 — 팀 공용 프레임워크(common.experiment) 인터페이스로 감싼다.

common.experiment.run_training 은 아래 두 함수를 정해진 시그니처로 부른다.
    state = fit(train_part, target_column, id_column)
    X     = transform(df, state, target_column, id_column)

fit 은 train_part(=fold 안) 에서만 호출되므로 상수열 제거(keep_idx)가 fold-safe 하다.
transform 은 한 행 안에서만 연산한다(부분집합 불변성). 그래서 Leakage 가 구조적으로 없다.

실제 피처 로직은 옆 모듈 features_A.py 에 있다. 여기서는 시그니처만 맞춘다.
탐색·ablation·CV·게이트는 exp 폴더의 pipeline.py 와 노트북이 담당한다.
"""
from __future__ import annotations

from .. import features_A as fa

# exp_002 에서 확정한 블록 조합. 바꾸려면 config.yaml 이 아니라 여기와 pipeline 을 함께 본다.
BLOCKS = ("G", "B", "V", "R")
SEED = 42


def fit(train_df, target_column, id_column) -> dict:
    gene_cols = [c for c in train_df.columns if c not in (target_column, id_column)]
    spec = fa.fit_spec(train_df, gene_cols, seed=SEED)
    return {"gene_cols": gene_cols, "spec": spec, "blocks": "".join(BLOCKS),
            "features_version": fa.__version__}


def transform(dataframe, state, target_column, id_column):
    del target_column, id_column                     # 행 안 연산이라 쓰지 않는다
    gene_cols = state["gene_cols"]
    counts = fa.parse_sample_counts(dataframe, gene_cols)
    X, _ = fa.build_features(dataframe, counts, state["spec"], tuple(state["blocks"]))
    return X                                          # scipy.sparse CSR — sklearn 선형 모델이 그대로 받는다
