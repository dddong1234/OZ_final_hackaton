from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from p1_eb_axes import EventDetail, parse_event_detail, summarize_ranks, aggregate_event_evidence, support_tables


class P1EbAxesTest(unittest.TestCase):
    def test_parser_keeps_wt_and_nan_out_of_event_space(self):
        for value in (None, np.nan, "WT", "", " nan "):
            event = parse_event_detail("TP53", value)
            self.assertIsNone(event.raw)
            self.assertEqual(event.event_type, "NONE")

    def test_parser_extracts_codon_and_allele(self):
        event = parse_event_detail("IDH1", "R132H")
        self.assertEqual(event.gene, "IDH1")
        self.assertEqual(event.event_type, "MISSENSE")
        self.assertEqual(event.position, 132)
        self.assertEqual(event.ref, "R")
        self.assertEqual(event.alt, "H")

    def test_rank_summary_respects_class_order(self):
        classes = np.asarray(["A", "B", "C"])
        labels = np.asarray(["B", "A"])
        probability = np.asarray([[0.2, 0.7, 0.1], [0.4, 0.3, 0.3]])
        summary, rows = summarize_ranks(probability, labels, classes)
        self.assertEqual(rows.true_rank.tolist(), [1, 1])
        self.assertEqual(rows.correct.tolist(), [True, True])
        self.assertEqual(float(summary.loc[0, "macro_recall_at_1"]), 1.0)
        self.assertEqual(float(summary.loc[0, "oracle_macro_f1_at_2"]), 1.0)

    def test_event_aggregates_return_sum_max_and_top2_per_class(self):
        event_scores = np.asarray([[1.0, -1.0], [2.0, 0.5], [-3.0, 3.0]], dtype=np.float32)
        result = aggregate_event_evidence(event_scores, n_classes=2)
        self.assertEqual(result.shape, (6,))
        np.testing.assert_allclose(result[:2], [0.0, 2.5], atol=1e-6)
        np.testing.assert_allclose(result[2:4], [2.0, 3.0], atol=1e-6)
        np.testing.assert_allclose(result[4:], [3.0, 3.5], atol=1e-6)

    def test_empty_structure_support_keeps_csv_schema(self):
        table = support_tables([[], []])
        self.assertEqual(table.columns.tolist(), ["gene", "event_count", "same_codon", "position_span"])


if __name__ == "__main__":
    unittest.main()
