import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from frozen_event_encoder import MODEL_ID, MODEL_REVISION, event_sentence, pool_event_embeddings


class FrozenEncoderTest(unittest.TestCase):
    def test_uses_public_renamed_biomedbert_with_pinned_revision(self):
        self.assertEqual(MODEL_ID, "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
        self.assertEqual(MODEL_REVISION, "2839b4fc440a3c41dc2b716fb14d530c33c8c1ff")

    def test_public_checkpoint_does_not_use_a_local_hub_token(self):
        runner = (Path(__file__).parent / "run_frozen_encoder.py").read_text(encoding="utf-8")
        self.assertIn('"token":False', runner)

    def test_blend_fold_metrics_are_recomputed_from_blend_probability(self):
        runner = (Path(__file__).parent / "run_frozen_encoder.py").read_text(encoding="utf-8")
        self.assertIn("blend_folds", runner)

    def test_sentence_is_fixed_and_uses_only_event_fields(self):
        self.assertEqual(event_sentence("TP53", "MISSENSE", "R", 175, "H"), "gene TP53 type MISSENSE ref R position 175 alt H")

    def test_empty_event_pool_is_zero_with_count_flag(self):
        pooled = pool_event_embeddings(np.empty((0, 2), dtype=np.float32), 2)
        np.testing.assert_allclose(pooled, [0, 0, 0, 0, 0])

    def test_pool_is_mean_max_and_log_count(self):
        pooled = pool_event_embeddings(np.asarray([[1, 3], [5, 2]], dtype=np.float32), 2)
        np.testing.assert_allclose(pooled, [3, 2.5, 5, 3, np.log1p(2)])


if __name__ == "__main__": unittest.main()
