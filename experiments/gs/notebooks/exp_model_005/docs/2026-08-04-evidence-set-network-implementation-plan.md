# Evidence Set Network Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** train-only raw profile audit와 seed42 Class-conditional Evidence Set Network screen을 `exp_model_005`에 추가한다.

**Architecture:** `team_ensemble_baseline.py`는 제공된 팀 3-way ensemble을 train-only vocabulary로 seed42 OOF 재현한다. `profile_audit.py`는 raw/normalized profile fingerprint와 purity를 계산한다. `evidence_set_core.py`는 fold-safe event evidence tensor, shared listwise scorer, metric helpers를 제공한다. `run_evidence_set_network.py`는 nested EB evidence와 outer OOF를 생성한다. 노트북은 실행·audit·시각화만 담당한다.

**Tech Stack:** Python 3.12, NumPy, pandas, SciPy CSR, scikit-learn, PyTorch, matplotlib, tqdm.

## Global Constraints

- 작업 범위는 `experiments/gs/notebooks/exp_model_005`만 수정한다.
- 팀 코드·기존 실험 파일을 import하지 않는다. 필요한 parser/EB/feature primitives는 이 디렉터리 안에 self-contained로 둔다.
- test 파일, test vocabulary, test statistics, submission 생성, external annotation을 screen에 사용하지 않는다.
- seed42, outer/inner Stratified 5-fold, hidden32, dropout0.15, AdamW lr0.001, weight decay0.0001, batch64, epoch60을 고정한다.
- P1+EB evidence는 outer-train 내부 inner OOF로 training row에 생성한다.
- team ensemble에 대한 OOF 재현이 `0.54202 ± 0.003` 허용 오차를 넘으면 candidate score를 판정하지 않고 `baseline_reproduction_match=false`로 저장한다.

---

### Task 1: Train-only team 3-way baseline reproduction

**Files:**
- Create: `common/team_ensemble_baseline.py`
- Create: `common/test_team_ensemble_baseline.py`

**Interfaces:**
- Produces: `build_train_only_gene_type_vocabulary(cache, fit_rows) -> tuple[list[str], dict[str, int]]`
- Produces: `project_gene_type_matrix(cache, rows, vocabulary) -> scipy.sparse.csr_matrix`
- Produces: `run_team_baseline_oof(train, genes, labels, seed) -> BaselineOOF`

- [ ] **Step 1: Write the failing train-only vocabulary test**

```python
def test_test_only_token_is_not_added_to_train_vocabulary():
    train_cache = FakeCache([("TP53", "MISSENSE")])
    valid_cache = FakeCache([("TEST_ONLY", "NONSENSE")])
    vocabulary, _ = build_train_only_gene_type_vocabulary(train_cache, np.array([0]))
    matrix = project_gene_type_matrix(valid_cache, np.array([0]), vocabulary)
    assert "TEST_ONLY__NONSENSE" not in vocabulary
    assert matrix.shape[1] == len(vocabulary)
    assert matrix.nnz == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m unittest common/test_team_ensemble_baseline.py`

Expected: FAIL because `team_ensemble_baseline` does not exist.

- [ ] **Step 3: Implement the self-contained baseline**

Copy only the user-provided deterministic parser, structured G+B+V+T+R+A+S feature construction, fold-train enrichment, and fixed models into this module. `run_team_baseline_oof` must create each outer fold's event vocabulary from that fold's training rows, project validation events onto it, preserve the label encoder class order, and blend `0.55 * multinomial + 0.30 * OVR + 0.15 * LightGBM` probabilities. It must return row-aligned `probabilities`, `fold_metrics`, `class_order`, and `audit`.

- [ ] **Step 4: Add reproduction guard**

```python
baseline_match = abs(oof_macro_f1 - 0.54202) <= 0.003
audit["baseline_reproduction_match"] = baseline_match
audit["test_read"] = False
```

- [ ] **Step 5: Run unit test and compile**

Run: `/Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m unittest common/test_team_ensemble_baseline.py && /Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m py_compile common/team_ensemble_baseline.py`

Expected: PASS.

### Task 2: Raw vs normalized profile audit

**Files:**
- Create: `common/profile_audit.py`
- Create: `common/test_profile_audit.py`
- Create: `common/run_raw_profile_audit.py`
- Create: `exp/exp-raw-profile-purity-audit-01.ipynb`

**Interfaces:**
- Produces: `raw_profile(frame, genes) -> list[str]`
- Produces: `normalized_profile(frame, genes) -> list[str]`
- Produces: `profile_purity(profile, labels) -> pandas.DataFrame`

- [ ] **Step 1: Write the failing test**

```python
def test_normalized_profile_collapses_case_prefix_and_delimiter_only():
    frame = pd.DataFrame({"G": ["p.R1H; P.Q2W", "R1H Q2W"]})
    assert normalized_profile(frame, ["G"])[0] == normalized_profile(frame, ["G"])[1]
    assert raw_profile(frame, ["G"])[0] != raw_profile(frame, ["G"])[1]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m unittest common/test_profile_audit.py`

Expected: FAIL because `profile_audit` does not exist.

- [ ] **Step 3: Implement minimal profile helpers and runner**

```python
def profile_purity(profile, labels):
    table = pd.DataFrame({"profile": profile, "label": labels})
    counts = table.groupby(["profile", "label"]).size().rename("n")
    group = counts.groupby(level=0).agg(["sum", "max"])
    group["purity"] = group["max"] / group["sum"]
    return group.reset_index()
```

The runner reads only `train.csv`, writes duplicate profile counts, weighted purity, per-class purity, and an audit JSON that records `test_read=false`.

- [ ] **Step 4: Run tests and compile**

Run: `/Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m unittest common/test_profile_audit.py && /Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m py_compile common/profile_audit.py common/run_raw_profile_audit.py`

Expected: PASS.

### Task 3: Fold-safe event evidence primitives

**Files:**
- Create: `common/evidence_set_core.py`
- Create: `common/test_evidence_set_core.py`

**Interfaces:**
- Produces: `event_evidence(events, weights, supports, row_ids, classes) -> list[np.ndarray]`
- Produces: `EvidenceSetNetwork(input_dim, hidden_dim=32, dropout=0.15)`
- Produces: `listwise_loss(logits, target_index, class_weight) -> torch.Tensor`

- [ ] **Step 1: Write the failing tests**

```python
def test_event_evidence_keeps_positive_and_negative_contributions():
    output = event_evidence(...)
    assert (output[0][:, 0] > 0).any()
    assert (output[0][:, 0] < 0).any()

def test_listwise_network_returns_one_score_per_class():
    logits = EvidenceSetNetwork(input_dim=14)(batch, mask)
    assert logits.shape == (2, 26)
```

- [ ] **Step 2: Run to verify failure**

Run: `/Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m unittest common/test_evidence_set_core.py`

Expected: FAIL because evidence helpers and model are missing.

- [ ] **Step 3: Implement fixed shared scorer**

```python
encoded = self.event_mlp(event_features)
mean = (encoded * mask[..., None]).sum(2) / mask.sum(2, keepdim=True).clamp_min(1)
maximum = encoded.masked_fill(~mask[..., None], -torch.inf).amax(2).nan_to_num(0)
summed = (encoded * mask[..., None]).sum(2)
return self.score(torch.cat([mean, maximum, summed], dim=-1)).squeeze(-1)
```

Use patient-level 26-way cross entropy; never flatten candidates into independent binary rows.

- [ ] **Step 4: Run tests**

Run: `/Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m unittest common/test_evidence_set_core.py`

Expected: PASS with finite scores and finite loss.

### Task 4: Nested OOF runner and safe notebook

**Files:**
- Create: `common/run_evidence_set_network.py`
- Create: `common/create_evidence_set_network_notebook.py`
- Create: `exp/exp-class-conditional-evidence-set-network-01.ipynb`
- Modify: `common/test_notebook_contract.py`

**Interfaces:**
- CLI: `--seed 42 --run-id exp-class-conditional-evidence-set-network-01`
- Produces: summary/fold/class/low-margin/OOF/loss/runtime/feature-contract/leakage-audit artifacts.

- [ ] **Step 1: Write a failing audit test**

```python
def test_nested_audit_rejects_validation_rows_in_eb_fit():
    audit = nested_audit(outer_train=np.array([0, 1]), inner_rows=np.array([0, 1]), outer_valid=np.array([2]))
    assert audit["outer_validation_used_for_eb_fit"] is False
```

- [ ] **Step 2: Run to verify failure**

Run: `/Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m unittest common/test_evidence_set_core.py`

Expected: FAIL because `nested_audit` is missing.

- [ ] **Step 3: Implement runner**

For every outer fold: generate inner OOF EB weights for outer train, generate outer-train EB weights for outer validation, train fixed network, align class order, save probabilities. The runner must call `run_team_baseline_oof` first. It records `baseline_reproduction_match`; if false, it writes ESN metrics but sets `promotion_decision="blocked_baseline_mismatch"`.

- [ ] **Step 4: Generate notebook**

The notebook defaults to `RUN_EXPERIMENT=False`, streams subprocess output with tqdm, asserts audit contracts, and plots Macro F1/fold delta/loss/low-margin results. It must not contain a literal `test.csv` string.

- [ ] **Step 5: Full static verification**

Run: `/Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m unittest discover -s common -p 'test_*.py' && /Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m py_compile common/team_ensemble_baseline.py common/profile_audit.py common/run_raw_profile_audit.py common/evidence_set_core.py common/run_evidence_set_network.py common/create_evidence_set_network_notebook.py && /Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python common/run_evidence_set_network.py --help`

Expected: all tests pass. Do not execute OOF during verification.

## Plan Self-Review

- Raw formatting audit and event-set model are separable and have independent tests.
- The team baseline is rebuilt with train-only vocabulary; it is not imported and it never combines train/validation caches before token encoding.
- Every supervised EB transformation is explicitly nested within outer training rows.
- The plan excludes test usage, hyperparameter search, threshold/blend search, and submission creation.
