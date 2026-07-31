from sklearn.linear_model import LogisticRegression


def build_model(config, seed):
    return LogisticRegression(
        max_iter=config["max_iter"],
        random_state=seed,
    )
