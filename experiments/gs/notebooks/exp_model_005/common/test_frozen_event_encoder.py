import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from frozen_event_encoder import event_sentence, pool_event_embeddings


class FrozenEncoderTest(unittest.TestCase):
    def test_sentence_is_fixed_and_uses_only_event_fields(self):
        self.assertEqual(event_sentence("TP53", "MISSENSE", "R", 175, "H"), "gene TP53 type MISSENSE ref R position 175 alt H")

    def test_empty_event_pool_is_zero_with_count_flag(self):
        pooled = pool_event_embeddings(np.empty((0, 2), dtype=np.float32), 2)
        np.testing.assert_allclose(pooled, [0, 0, 0, 0, 0])

    def test_pool_is_mean_max_and_log_count(self):
        pooled = pool_event_embeddings(np.asarray([[1, 3], [5, 2]], dtype=np.float32), 2)
        np.testing.assert_allclose(pooled, [3, 2.5, 5, 3, np.log1p(2)])


if __name__ == "__main__": unittest.main()
