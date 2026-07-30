"""팀 방식 학습 진입점 — python -m experiments.member_d.exp_002_variant_type.training.run

common.experiment.run_training 이 holdout(baseline.yaml test_size 0.25) 로 seed 별 학습 후
확률 평균 앙상블하고, results/metrics.json · submission.csv · model.joblib 을 남긴다.
탐색·CV·게이트는 exp 폴더의 pipeline.py 를 쓴다.
"""
from common.experiment import run_training

if __name__ == "__main__":
    _, member, experiment, _ = __package__.split(".")
    run_training(member, experiment)
