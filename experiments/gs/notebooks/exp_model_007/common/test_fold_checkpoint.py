import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from fold_checkpoint import experiment_result_dir, load_checkpoint, save_checkpoint  # noqa: E402


class FoldCheckpointTest(unittest.TestCase):
    def test_result_dir_is_the_experiment_result_sibling_of_common(self):
        runner = Path("/repo/experiments/gs/notebooks/exp_model_007/common/run.py")
        self.assertEqual(
            experiment_result_dir(runner),
            Path("/repo/experiments/gs/notebooks/exp_model_007/result"),
        )

    def test_checkpoint_round_trip_preserves_completed_folds_and_oof_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_seed42_checkpoint.npz"
            payload = {
                "completed_folds": [1, 3],
                "fold_rows": [{"fold": 1, "variant": "H0", "macro_f1": 0.5}],
                "audit_rows": [{"fold": 1, "leakage_check": True}],
                "oof": {"h0": np.asarray([[.2, .8], [.7, .3]], dtype=np.float32)},
            }
            save_checkpoint(path, payload)
            restored = load_checkpoint(path)
            self.assertEqual(restored["completed_folds"], [1, 3])
            self.assertEqual(restored["fold_rows"], payload["fold_rows"])
            self.assertEqual(restored["audit_rows"], payload["audit_rows"])
            np.testing.assert_allclose(restored["oof"]["h0"], payload["oof"]["h0"])


if __name__ == "__main__":
    unittest.main()
