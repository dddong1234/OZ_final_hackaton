import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from raw_canonical_audit import build_profiles, event_type, split_events  # noqa: E402


class RawCanonicalAuditTest(unittest.TestCase):
    def test_nan_wt_and_blank_produce_no_events(self):
        self.assertEqual(split_events(float("nan")), ())
        self.assertEqual(split_events("WT"), ())
        self.assertEqual(split_events("  "), ())

    def test_canonical_profile_splits_delimited_events(self):
        frame = pd.DataFrame({"TP53": ["p.R175H; R248Q"]})
        profiles, audit = build_profiles(frame, ["TP53"])
        self.assertEqual(profiles["canonical_event"][0], "TP53=R175H|TP53=R248Q")
        self.assertEqual(audit["raw_segment_count"], 2)
        self.assertTrue(audit["segment_conservation"])

    def test_gene_type_collapses_events_but_retains_event_profile(self):
        frame = pd.DataFrame({"TP53": ["R175H R248Q"]})
        profiles, _ = build_profiles(frame, ["TP53"])
        self.assertEqual(profiles["canonical_event"][0], "TP53=R175H|TP53=R248Q")
        self.assertEqual(profiles["gene_type"][0], "TP53__MISSENSE")
        self.assertEqual(event_type("R175H"), "MISSENSE")


if __name__ == "__main__":
    unittest.main()
