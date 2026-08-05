import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from auto_validated_pair_specialist import (  # noqa: E402
    apply_pair_probability,
    select_non_overlapping_pairs,
    top_two_pair_mask,
)


class AutoValidatedPairSpecialistTest(unittest.TestCase):
    def test_pair_route_requires_exact_h0_top_two_pair(self):
        classes = np.array(["A", "B", "C"])
        probability = np.array([[.55, .40, .05], [.70, .10, .20]])
        routed = top_two_pair_mask(probability, classes, ("A", "B"))
        np.testing.assert_array_equal(routed, [True, False])

    def test_probability_mass_is_preserved_inside_pair(self):
        classes = np.array(["A", "B", "C"])
        base = np.array([[.50, .30, .20]])
        output = apply_pair_probability(base, np.array([[.10, .90]]), classes, ("A", "B"))
        np.testing.assert_allclose(output.sum(axis=1), 1.0)
        np.testing.assert_allclose(output[0, :2].sum(), .80)
        np.testing.assert_allclose(output[0, 2], .20)

    def test_selected_pairs_do_not_share_class(self):
        candidates = [
            {"pair": ("A", "B"), "pair_f1_delta": .10, "recovered": 8, "broken": 1},
            {"pair": ("A", "C"), "pair_f1_delta": .09, "recovered": 9, "broken": 0},
            {"pair": ("D", "E"), "pair_f1_delta": .08, "recovered": 3, "broken": 0},
        ]
        selected = select_non_overlapping_pairs(candidates, maximum=2)
        self.assertEqual(selected, [("A", "B"), ("D", "E")])


if __name__ == "__main__":
    unittest.main()
