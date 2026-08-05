# Raw Canonical Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine, using train data only, whether raw mutation notation retains label-relevant information that H0's gene-by-functional-type canonicalization discards.

**Architecture:** A self-contained parser produces three deterministic train-row fingerprints: raw cell text, canonical event tokens, and H0-compatible gene-by-type tokens. The runner compares duplicate-profile purity and profile disagreements without fitting a classifier or reading test data.

**Tech Stack:** Python 3.12, pandas, NumPy, matplotlib, unittest, tqdm.

## Global Constraints

- Write only under `experiments/gs/notebooks/exp_model_011`.
- Read `data/raw/train.csv` only; do not read `test.csv`.
- No external data, fixed cancer/gene/mutation lists, or label-derived input rules.
- NaN, WT, and blanks produce zero events.
- Do not fit a model or create a submission in this audit.

---

### Task 1: Canonical profile library and parser-contract tests

**Files:**
- Create: `common/raw_canonical_audit.py`
- Create: `common/test_raw_canonical_audit.py`

**Interfaces:**
- Produces `split_events(value) -> tuple[str, ...]`, `event_type(event) -> str`, and `build_profiles(frame, genes) -> dict[str, list[str]]`.
- The runner consumes the three lists named `raw`, `canonical_event`, and `gene_type`.

- [ ] **Step 1: Write failing parser tests**

```python
def test_nan_wt_and_blank_produce_no_events():
    assert split_events(float('nan')) == ()
    assert split_events('WT') == ()
    assert split_events(' ') == ()

def test_canonical_profile_splits_delimited_events():
    frame = pd.DataFrame({'TP53': ['p.R175H; R248Q']})
    assert build_profiles(frame, ['TP53'])['canonical_event'][0] == 'TP53=R175H|TP53=R248Q'
```

- [ ] **Step 2: Run tests and verify expected failure**

Run: `python common/test_raw_canonical_audit.py`

- [ ] **Step 3: Implement deterministic parser and profiles**

```python
def split_events(value):
    if pd.isna(value):
        return ()
    text = str(value).strip().upper()
    if not text or text == 'WT':
        return ()
    return tuple(dict.fromkeys(token.removeprefix('P.') for token in re.sub(r'[;,|]+', ' ', text).split() if token))
```

- [ ] **Step 4: Re-run parser tests**

Run: `python common/test_raw_canonical_audit.py`

### Task 2: Train-only audit runner and schema tests

**Files:**
- Create: `common/run_raw_canonical_audit.py`
- Create: `common/test_run_raw_canonical_audit.py`

**Interfaces:**
- CLI: `python common/run_raw_canonical_audit.py --run-id exp-raw-canonical-audit-01 [--smoke]`.
- Produces summary, purity, disagreement CSVs and audit JSON in `result/`.

- [ ] **Step 1: Write failing output-schema test**

```python
def test_summary_schema_contains_all_profile_kinds(tmp_path):
    summary = summarize_profiles(['A', 'B'], ['A', 'C'], ['x', 'y'], np.array(['X', 'Y']))
    assert set(summary.profile_kind) == {'raw', 'canonical_event', 'gene_type'}
```

- [ ] **Step 2: Run test and verify expected failure**

Run: `python common/test_run_raw_canonical_audit.py`

- [ ] **Step 3: Implement audit runner**

It must assert train-only operation, preserve every non-WT raw segment, record parser coverage, calculate duplicate-profile purity, and write only non-sensitive aggregate/profile identifiers.

- [ ] **Step 4: Re-run schema and smoke tests**

Run: `python common/test_run_raw_canonical_audit.py && python common/run_raw_canonical_audit.py --smoke`

### Task 3: Notebook, documentation, and static verification

**Files:**
- Create: `exp/exp-raw-canonical-audit-01.ipynb`
- Create: `docs/2026-08-05-raw-canonical-audit.md`

**Interfaces:**
- Notebook invokes the runner, loads all CSVs safely, renders purity/disagreement plots, and prints an explicit go/no-go verdict for raw token modeling.

- [ ] **Step 1: Create notebook with run switch and tqdm subprocess output**

Use `RUN_EXPERIMENT=False` by default and run only `train.csv` through the CLI.

- [ ] **Step 2: Add result guards**

```python
assert audit['test_read'] is False
assert audit['nan_as_mutation_count'] == 0
assert audit['segment_conservation'] is True
```

- [ ] **Step 3: Verify Python and notebook structure**

Run: `python -m py_compile common/raw_canonical_audit.py common/run_raw_canonical_audit.py && python -m json.tool exp/exp-raw-canonical-audit-01.ipynb >/dev/null`

- [ ] **Step 4: Commit**

```bash
git add experiments/gs/notebooks/exp_model_011
git commit -m "feat(gs): add raw canonical mutation audit"
```

## Self-Review

- Scope is audit-only; no classifier, blend, threshold, or submission is implemented.
- Raw versus canonical distinctions are explicit and use no test statistics.
- All parser and output-contract checks are executable before a full audit run.
