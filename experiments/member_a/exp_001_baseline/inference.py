from common.experiment import run_inference


if __name__ == "__main__":
    _, member, experiment = __package__.split(".")
    run_inference(member, experiment)
