import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from h2_evidence_shape_core import (  # noqa: E402
    assert_fold_contract,
    parse_frame,
    fit_event_state,
    transform_event_state,
)


class H2EvidenceShapeCoreTest(unittest.TestCase):
    def test_nan_wt_and_blank_do_not_create_events(self):
        frame = pd.DataFrame({"G1": ["WT", np.nan, "", "R1H"], "G2": [" ", "WT", None, "R2*"]})
        parsed = parse_frame(frame, ["G1", "G2"])
        self.assertEqual(len(parsed.events), 2)
        self.assertEqual(parsed.nan_as_mutation_count, 0)

    def test_validation_only_event_is_not_added_to_fit_vocabulary(self):
        fit = pd.DataFrame({"G1": ["R1H"]})
        valid = pd.DataFrame({"G1": ["Q2W"]})
        state = fit_event_state(parse_frame(fit, ["G1"]), np.asarray(["A"]))
        transformed = transform_event_state(parse_frame(valid, ["G1"]), state)
        self.assertEqual(state.exact_vocabulary, ("G1__R1H",))
        self.assertEqual(transformed.exact.shape[1], 1)
        self.assertEqual(transformed.exact.nnz, 0)

    def test_outer_validation_is_not_in_any_fit_partition(self):
        audit = assert_fold_contract(np.asarray([0, 1, 2]), np.asarray([0, 1]), np.asarray([3]))
        self.assertTrue(audit["leakage_check"])
        self.assertFalse(audit["outer_validation_used_for_fit"])


if __name__ == "__main__":
    unittest.main()
