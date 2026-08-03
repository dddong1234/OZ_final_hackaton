from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from parser_recovery import event_tokens, parse_cell


class ParserRecoveryTest(unittest.TestCase):
    def test_wt_and_nan_are_not_events(self):
        for value in (None, np.nan, "", " WT ", "nan"):
            self.assertEqual(parse_cell("TP53", value), [])

    def test_multi_event_segments_are_unique(self):
        events = parse_cell("TP53", "R175H; R248Q / R175H")
        self.assertEqual([event.raw for event in events], ["R175H", "R248Q"])

    def test_whitespace_separated_events_are_recovered(self):
        events = parse_cell("BRAF", "L26V L24V")
        self.assertEqual([event.raw for event in events], ["L26V", "L24V"])

    def test_canonical_types_and_unknown_retention(self):
        self.assertEqual(parse_cell("BRAF", "V600E")[0].canonical_type, "MISSENSE")
        self.assertEqual(parse_cell("APC", "R1450*")[0].canonical_type, "NONSENSE")
        self.assertEqual(parse_cell("BRCA1", "K123fs")[0].canonical_type, "FRAMESHIFT_DEL")
        self.assertEqual(parse_cell("X", "483_484MP>IA")[0].canonical_type, "DELINS_COMPLEX")
        self.assertEqual(parse_cell("X", "mystery_format")[0].canonical_type, "UNKNOWN")

    def test_g1_uses_legacy_parent_and_g2_keeps_canonical(self):
        rows = [parse_cell("BRCA1", "K123fs")]
        self.assertEqual(event_tokens(rows, "legacy")[0], {"BRCA1__FRAMESHIFT"})
        self.assertEqual(event_tokens(rows, "canonical")[0], {"BRCA1__FRAMESHIFT_DEL"})


if __name__ == "__main__":
    unittest.main()
