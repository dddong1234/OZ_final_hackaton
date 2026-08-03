"""Generate the fixed exp13 submission: 0.5 primary LR + 0.5 event-token OVR LR.

Only train.csv is used to fit feature rules, TF-IDF vocabulary/IDF, and models.
test.csv is transformed once; it is never used for fit, statistics, or selection.
"""
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
RUNNER = ROOT / "experiments" / "gs" / "notebooks" / "eda_pre_002" / "common" / "exp-gs-002-memory-safe.py"
CANDIDATE = "H-AS-LR-exact-confusion-pairs-Apair-log1p"
SEED = 42
MIN_DF = 3
SUBMISSION_NAME = "submission_exp-gs-002-13_primary-ovr-tfidf_blend_seed42.csv"


if __name__ == "__main__":
    subprocess.run([
        sys.executable, str(RUNNER),
        "--submit",
        "--submit-event-tfidf-ovr",
        "--candidate", CANDIDATE,
        "--model", "logistic",
        "--seed", str(SEED),
        "--tfidf-min-df", str(MIN_DF),
        "--submission-name", SUBMISSION_NAME,
    ], check=True, cwd=ROOT)
