from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from h0_selective_eb_submission import make_submission_frame, submission_directory


class SubmissionContractTest(unittest.TestCase):
    def test_submission_preserves_sample_id_order_and_schema(self) -> None:
        sample = pd.DataFrame({"ID": ["t2", "t1"], "SUBCLASS": ["", ""]})
        test = pd.DataFrame({"ID": ["t2", "t1"]})
        probability = np.asarray([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
        output = make_submission_frame(sample, test, probability, np.asarray(["A", "B"], dtype=object))
        self.assertEqual(list(output.columns), ["ID", "SUBCLASS"])
        self.assertEqual(output.ID.tolist(), ["t2", "t1"])
        self.assertEqual(output.SUBCLASS.tolist(), ["B", "A"])

    def test_submission_rejects_mismatched_ids(self) -> None:
        sample = pd.DataFrame({"ID": ["t1"], "SUBCLASS": [""]})
        test = pd.DataFrame({"ID": ["wrong"]})
        with self.assertRaises(ValueError):
            make_submission_frame(sample, test, np.asarray([[1.0]], dtype=np.float32), np.asarray(["A"], dtype=object))

    def test_submission_output_stays_in_gs_submission_directory(self) -> None:
        path = submission_directory()
        self.assertEqual(path.name, "submission")
        self.assertIn("experiments/gs/notebooks", str(path))


if __name__ == "__main__":
    unittest.main()
