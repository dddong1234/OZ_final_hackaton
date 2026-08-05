import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from nb_profile_blend import build_gene_type_matrix, fixed_blend  # noqa: E402


class NBProfileBlendTest(unittest.TestCase):
    def test_nan_wt_and_blank_do_not_create_profile_tokens(self):
        frame = pd.DataFrame({"G1": ["WT", np.nan, "", "R1H"], "G2": [" ", "WT", None, "R2*"]})
        matrix, vocabulary = build_gene_type_matrix(frame, ["G1", "G2"], vocabulary=None)
        self.assertEqual(matrix.shape[0], 4)
        self.assertEqual(matrix.getnnz(), 2)
        self.assertEqual(len(vocabulary), 2)

    def test_validation_only_token_is_ignored_by_train_vocabulary(self):
        train = pd.DataFrame({"G1": ["R1H"], "G2": ["WT"]})
        valid = pd.DataFrame({"G1": ["WT"], "G2": ["R2*"]})
        _, vocabulary = build_gene_type_matrix(train, ["G1", "G2"], vocabulary=None)
        matrix, _ = build_gene_type_matrix(valid, ["G1", "G2"], vocabulary=vocabulary)
        self.assertEqual(matrix.nnz, 0)

    def test_fixed_blend_preserves_probability_normalization(self):
        h0 = np.asarray([[0.8, 0.2], [0.3, 0.7]])
        nb = np.asarray([[0.1, 0.9], [0.6, 0.4]])
        result = fixed_blend(h0, nb)
        np.testing.assert_allclose(result.sum(axis=1), 1.0)
        np.testing.assert_allclose(result, 0.75 * h0 + 0.25 * nb)


if __name__ == "__main__":
    unittest.main()
