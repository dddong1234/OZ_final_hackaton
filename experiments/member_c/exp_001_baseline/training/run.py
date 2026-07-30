from common.experiment import run_training


if __name__ == "__main__":
    _, member, experiment, _ = __package__.split(".")
    run_training(member, experiment)
