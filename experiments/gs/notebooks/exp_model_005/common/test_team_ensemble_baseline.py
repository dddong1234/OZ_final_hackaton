import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from team_ensemble_baseline import (  # noqa: E402
    EventCache,
    build_train_only_gene_type_vocabulary,
    parse_train_frame,
    project_gene_type_matrix,
)


class TrainOnlyVocabularyTest(unittest.TestCase):
    def test_validation_only_token_is_not_added_to_train_vocabulary(self):
        cache = EventCache.from_rows(
            [[("TP53", "MISSENSE")], [("VALID_ONLY", "NONSENSE")]]
        )
        vocabulary = build_train_only_gene_type_vocabulary(cache, np.array([0]))
        projected = project_gene_type_matrix(cache, np.array([1]), vocabulary)

        self.assertEqual(vocabulary, ("TP53__MISSENSE",))
        self.assertEqual(projected.shape, (1, 1))
        self.assertEqual(projected.nnz, 0)

    def test_parser_excludes_wt_blank_and_nan_from_events(self):
        frame = pd.DataFrame({"TP53": ["WT", "", np.nan, "p.R175H"]})
        cache = parse_train_frame(frame, ["TP53"], show_progress=False)
        self.assertEqual(cache.events[["row", "event"]].to_dict("records"), [{"row": 3, "event": "R175H"}])


if __name__ == "__main__":
    unittest.main()
