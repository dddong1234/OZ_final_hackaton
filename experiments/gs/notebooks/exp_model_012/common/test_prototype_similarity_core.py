import numpy as np
import pandas as pd
import unittest
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


class PrototypeSimilarityCoreTest(unittest.TestCase):
    def test_nan_wt_and_blank_are_not_event_tokens(self):
        from prototype_similarity_core import parse_event_tokens

        frame = pd.DataFrame({"G": [np.nan, "WT", "", "R132H"]})
        self.assertEqual(parse_event_tokens(frame, ["G"]), [[], [], [], ["G__MISSENSE", "G__R132H"]])

    def test_train_only_prototype_probability_is_row_normalized(self):
        from prototype_similarity_core import fit_train_only_prototype, predict_prototype

        frame = pd.DataFrame({"G": ["R132H", "WT", "V600E", "WT"]})
        labels = np.asarray(["A", "A", "B", "B"], dtype=object)
        classes = np.asarray(["A", "B"], dtype=object)
        artifacts = fit_train_only_prototype(frame, labels, ["G"], classes)
        probability = predict_prototype(frame, ["G"], artifacts)

        self.assertEqual(probability.shape, (4, 2))
        np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-7)

    def test_fit_tokens_do_not_expand_for_apply_only_event(self):
        from prototype_similarity_core import fit_train_only_prototype, predict_prototype

        fit = pd.DataFrame({"G": ["R132H", "WT"]})
        apply = pd.DataFrame({"G": ["V600E"]})
        artifacts = fit_train_only_prototype(fit, np.asarray(["A", "B"], dtype=object), ["G"], np.asarray(["A", "B"], dtype=object))

        self.assertNotIn("G__V600E", artifacts.vocabulary)
        np.testing.assert_allclose(predict_prototype(apply, ["G"], artifacts).sum(axis=1), 1.0, atol=1e-7)
