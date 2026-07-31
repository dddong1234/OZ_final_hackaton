"""팀 방식 추론 진입점 — python -m experiments.iljun.exp_002_variant_type.inference

저장된 model.joblib 로 test 를 예측해 results/submission.csv 를 만든다.
"""
from common.experiment import run_inference

if __name__ == "__main__":
    _, member, experiment = __package__.split(".")
    run_inference(member, experiment)
