import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from profile_retrieval import (  # noqa: E402
    build_profile_lookup,
    fixed_profile_blend,
    profile_key,
    query_profile_posteriors,
)


class ProfileRetrievalTest(unittest.TestCase):
    def test_nan_wt_and_blank_profiles_do_not_become_events(self):
        row = pd.Series({"G1": np.nan, "G2": "WT", "G3": "  "})
        self.assertEqual(profile_key(row, ["G1", "G2", "G3"]), "")

    def test_lookup_uses_fit_labels_only(self):
        fit = pd.DataFrame({"G1": ["A1V", "A1V"], "G2": ["WT", "WT"]})
        labels = np.array(["A", "A"])
        lookup = build_profile_lookup(fit, labels, ["G1", "G2"], np.array(["A", "B"]))
        query = pd.DataFrame({"G1": ["A1V"], "G2": ["WT"]})
        posterior, matched, support, purity = query_profile_posteriors(query, ["G1", "G2"], lookup)
        self.assertTrue(matched[0])
        self.assertEqual(int(support[0]), 2)
        self.assertAlmostEqual(float(purity[0]), 1.0)
        self.assertGreater(posterior[0, 0], posterior[0, 1])

    def test_production_code_never_reads_test_or_joins_train_test(self):
        source = "\n".join(
            (Path(__file__).parent / name).read_text()
            for name in ("profile_retrieval.py", "run_profile_retrieval.py")
            if (Path(__file__).parent / name).exists()
        )
        self.assertNotIn("test.csv", source)
        self.assertNotIn("pd.concat", source)
        self.assertNotIn("FINAL_EXACT", source)

    def test_fixed_blend_only_changes_matched_rows(self):
        h0 = np.array([[.70, .30], [.20, .80]])
        profile = np.array([[.10, .90], [.90, .10]])
        blended = fixed_profile_blend(h0, profile, np.array([True, False]))
        np.testing.assert_allclose(blended[0], [.58, .42])
        np.testing.assert_allclose(blended[1], h0[1])


if __name__ == "__main__":
    unittest.main()
