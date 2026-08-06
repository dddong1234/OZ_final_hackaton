import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from run_raw_canonical_audit import summarize_profiles  # noqa: E402


class RawCanonicalRunnerTest(unittest.TestCase):
    def test_summary_schema_contains_all_profile_kinds(self):
        summary, details = summarize_profiles(
            {"raw": ["A", "B"], "canonical_event": ["A", "C"], "gene_type": ["x", "y"]},
            np.array(["X", "Y"]),
        )
        self.assertEqual(set(summary.profile_kind), {"raw", "canonical_event", "gene_type"})
        self.assertEqual(set(details), {"raw", "canonical_event", "gene_type"})


if __name__ == "__main__":
    unittest.main()
