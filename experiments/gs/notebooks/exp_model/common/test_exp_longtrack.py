"""Contracts for input audit and NB-ratio OVR runners."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
from scipy import sparse

def load(name):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec); assert spec and spec.loader
    sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module

AUDIT = load("exp_input_audit_runner.py")
RATIO = load("exp_nb_ratio_ovr_runner.py")

def test_full_event_profile_preserves_gene_event_pairs() -> None:
    assert AUDIT.canonical_profile([("TP53", "R175H"), ("PIK3CA", "E545K")]) == "PIK3CA=E545K|TP53=R175H"

def test_nearest_metrics_ignore_self_and_return_exact_jaccard() -> None:
    matrix = sparse.csr_matrix([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.float32)
    nearest = AUDIT.nearest_profile_metrics(matrix, block_size=2)
    assert nearest.loc[0, "jaccard_neighbor"] == 1
    assert nearest.loc[0, "jaccard_similarity"] == 1.0

def test_nb_ratio_is_fold_train_class_conditional() -> None:
    matrix = sparse.csr_matrix([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    target = np.array([1, 1, 0, 0])
    ratio = RATIO.log_count_ratio(matrix, target, smoothing=1.0)
    assert ratio[0] > 0 and ratio[1] < 0

def test_nb_ratio_rejects_signed_features() -> None:
    matrix = sparse.csr_matrix([[1, -1], [0, 1]], dtype=np.float32)
    try:
        RATIO.log_count_ratio(matrix, np.array([1, 0]), smoothing=1.0)
    except ValueError as error:
        assert "nonnegative" in str(error)
    else:
        raise AssertionError("signed NB-ratio input must be rejected")

def test_longtrack_runners_never_reference_test_csv() -> None:
    for name in ("exp_input_audit_runner.py", "exp_nb_ratio_ovr_runner.py"):
        assert '"test.csv"' not in Path(__file__).with_name(name).read_text(encoding="utf-8")

if __name__ == "__main__":
    test_full_event_profile_preserves_gene_event_pairs(); test_nearest_metrics_ignore_self_and_return_exact_jaccard()
    test_nb_ratio_is_fold_train_class_conditional(); test_nb_ratio_rejects_signed_features(); test_longtrack_runners_never_reference_test_csv()
    print("Long-track contracts passed")
