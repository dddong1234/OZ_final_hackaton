import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE = Path(__file__).with_name("h0_complement_nb_profile.py")
SPEC = importlib.util.spec_from_file_location("h0_complement_nb_profile", MODULE)
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


class ProfileBlendTest(unittest.TestCase):
    def test_profile_blend_uses_fixed_weights_and_preserves_probability_rows(self):
        h0 = np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=np.float32)
        nb = np.asarray([[0.2, 0.8], [0.6, 0.4]], dtype=np.float32)
        actual = core.profile_blend(h0, nb)
        np.testing.assert_allclose(actual, 0.8 * h0 + 0.2 * nb)
        np.testing.assert_allclose(actual.sum(axis=1), 1.0)

    def test_profile_blend_rejects_probability_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "share shape"):
            core.profile_blend(np.ones((2, 2)) / 2, np.ones((2, 3)) / 3)


if __name__ == "__main__":
    unittest.main()
