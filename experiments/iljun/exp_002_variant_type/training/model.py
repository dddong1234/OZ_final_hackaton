"""
트랙 A 모델 — common.experiment 이 build_model(config, seed) 로 부른다.

class_weight="balanced" 는 이 데이터에서 사실상 필수다 (참고사항 3):
제거하면 Macro F1 −0.032 / Accuracy −0.009. 불균형 20.7배라 소수 클래스가 사라진다.
common.experiment 은 predict_proba 로 seed 앙상블을 하므로 확률을 내는 모델이어야 한다
(LinearSVC 는 predict_proba 가 없어 이 프레임워크에서 못 쓴다 → LogisticRegression).
"""
from sklearn.linear_model import LogisticRegression


def build_model(config, seed):
    return LogisticRegression(
        max_iter=config.get("max_iter", 1000),
        class_weight=config.get("class_weight", "balanced"),
        random_state=seed,
    )
