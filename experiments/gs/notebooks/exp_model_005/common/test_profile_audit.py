import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from profile_audit import normalized_profile, raw_profile  # noqa: E402


class ProfileAuditTest(unittest.TestCase):
    def test_normalized_profile_collapses_case_prefix_and_delimiter_only(self):
        frame = pd.DataFrame({"TP53": ["p.R1H; P.Q2W", "R1H Q2W"]})
        normalized = normalized_profile(frame, ["TP53"])
        raw = raw_profile(frame, ["TP53"])

        self.assertEqual(normalized[0], normalized[1])
        self.assertNotEqual(raw[0], raw[1])


if __name__ == "__main__":
    unittest.main()
