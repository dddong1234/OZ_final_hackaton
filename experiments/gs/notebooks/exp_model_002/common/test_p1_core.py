"""P1 독립 공통 코드의 최소 계약 테스트."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from p1_core import event_tokens, parse_event, is_mutation, fit_log_odds, apply_log_odds, build_cache, recurrent_matrix


def test_event_contract():
    assert not is_mutation(np.nan)
    assert not is_mutation('WT')
    assert not is_mutation('  ')
    assert is_mutation('R132H')
    assert parse_event('R132H')['type'] == 'MISSENSE'
    assert parse_event('E746_A750del')['type'] == 'INFRAME_DEL'
    assert parse_event('R97*')['type'] == 'NONSENSE'


def test_tokens_and_scores_are_train_fitted():
    frame = pd.DataFrame({'G1': ['R132H', 'WT', np.nan], 'G2': ['WT', 'R97*', 'A10V']})
    toks = event_tokens(frame)
    assert toks[0] == {'G1__MISSENSE'}
    assert toks[1] == {'G2__NONSENSE'}
    assert toks[2] == {'G2__MISSENSE'}
    classes = np.array(['A', 'B'])
    weights = fit_log_odds(toks[:2], np.array(['A', 'B']), classes, min_support=1)
    scored = apply_log_odds(toks, weights, classes)
    assert scored.shape == (3, 2)
    assert np.isfinite(scored).all()


def test_recurrent_vocab_column_indices_are_dense():
    frame = pd.DataFrame({
        'G0': ['R1A', 'WT', 'WT', 'WT', 'WT', 'WT'],
        'G1': ['WT', 'R1A', 'WT', 'WT', 'WT', 'WT'],
        'G2': ['WT', 'WT', 'R1A', 'WT', 'WT', 'WT'],
        'G3': ['WT', 'WT', 'WT', 'R1A', 'WT', 'WT'],
        'G4': ['WT', 'WT', 'WT', 'WT', 'R1A', 'WT'],
        'ZZ': ['A2V'] * 5 + ['WT'],
    })
    cache = build_cache(frame)
    x = recurrent_matrix(cache, np.arange(6), np.arange(6), threshold=5)
    assert x.shape == (6, 1)
    assert x[:, 0].sum() == 5


def test_legacy_p1_reference_contract_is_available():
    from legacy_p1_reference import load_reference
    base, enrichment = load_reference()
    assert callable(base._matrix)
    assert callable(enrichment.cross_fitted_enrichment)


if __name__ == '__main__':
    test_event_contract(); test_tokens_and_scores_are_train_fitted(); test_recurrent_vocab_column_indices_are_dense(); test_legacy_p1_reference_contract_is_available()
    print('p1_core contract tests passed')
