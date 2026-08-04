import numpy as np
from scipy import sparse
import inspect

import feature_experiment as exp


def _fold():
    names = ["G__A", "G__B", "B__event_count", "V__missense_event_count", "E__gene_type__X"]
    fit = sparse.csr_matrix([[1, 0, 0, 0, -1], [1, 0, np.log1p(2), np.log1p(1), 1], [0, 1, np.log1p(8), np.log1p(3), 0]], dtype=np.float32)
    valid = sparse.csr_matrix([[0, 1, np.log1p(1), np.log1p(1), 0]], dtype=np.float32)
    return exp.PreparedFold(1, np.arange(3), np.array([0]), np.array(["X", "X", "Y"]), np.array(["Y"]), fit, valid, names)


def test_gene_support_uses_fit_only():
    fold = _fold()
    case = exp.FeatureCase("support2", ("G__",), gene_min_support=2)
    mask = exp._column_mask(fold, case)
    assert mask.tolist() == [True, False, False, False, False]


def test_fixed_bins_have_fixed_shape_and_are_row_local():
    fold = _fold()
    fit_bins = exp._fixed_bins(fold.x_fit, fold.names)
    valid_bins = exp._fixed_bins(fold.x_valid, fold.names)
    assert fit_bins.shape == (3, 12)
    assert valid_bins.shape == (1, 12)
    assert np.all(np.asarray(fit_bins.sum(axis=1)).ravel() == 2)


def test_case_catalog_is_unique_and_grouped():
    assert len(exp.CASES) == 26
    assert len(exp.CASE_MAP) == len(exp.CASES)
    grouped = sum((exp.cases_for_group(name) for name in exp.GROUPS), [])
    assert {case.name for case in grouped} == set(exp.CASE_MAP)


def test_probability_blend_endpoints():
    fold = _fold()
    fold.valid_index = np.array([0])
    fold.y_valid = np.array(["Y"])
    lr = {"oof_f1_macro": 0.0, "classes": np.array(["X", "Y"]), "prediction": np.array(["X"]), "probabilities": np.array([[0.8, 0.2]])}
    lgbm = {"oof_f1_macro": 1.0, "classes": np.array(["X", "Y"]), "prediction": np.array(["Y"]), "probabilities": np.array([[0.1, 0.9]])}
    table = exp.search_lr_blends([fold], {"candidate": lgbm}, lr, ["candidate"], lgbm_weights=(0.0, 1.0))
    assert table.iloc[0].lgbm_weight == 1.0
    assert table.iloc[0].oof_f1_macro == 1.0


def test_catalog_contains_no_fixed_domain_blocks():
    for case in exp.CASES:
        prefixes = case.include_prefixes or ()
        assert "C__" not in prefixes
        assert "D__" not in prefixes
        assert not any(name.startswith(("C__", "D__")) for name in case.include_names)
    source = inspect.getsource(exp.prepare_seed)
    assert "use_fixed_contrast=False" in source
    assert 'name.startswith(("C__", "D__exact_"))' in source
