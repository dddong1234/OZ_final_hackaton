"""exp-input-audit-01: train-only full-event identifiability audit."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm.auto import tqdm
import sparse_fm_runner as base

def canonical_profile(events: list[tuple[str, str]]) -> str:
    return "|".join(f"{gene}={event}" for gene, event in sorted(set(events)))

def event_matrix(cache: base.Cache) -> sparse.csr_matrix:
    if cache.events.empty: return sparse.csr_matrix((cache.mutation.shape[0], 0), dtype=np.float32)
    token = cache.events.gene + "=" + cache.events.event
    names = sorted(token.unique()); lookup = {name: i for i, name in enumerate(names)}
    return sparse.coo_matrix((np.ones(len(token), np.float32), (cache.events.row, token.map(lookup))), shape=(cache.mutation.shape[0], len(names))).tocsr()

def nearest_profile_metrics(matrix: sparse.csr_matrix, block_size: int = 128) -> pd.DataFrame:
    sizes = np.asarray(matrix.getnnz(axis=1)).ravel().astype(np.float32); norm = np.sqrt(np.maximum(sizes, 1))
    neighbor_j = np.full(matrix.shape[0], -1, dtype=int); cosine = np.zeros(matrix.shape[0], np.float32); jaccard = np.zeros(matrix.shape[0], np.float32)
    for start in tqdm(range(0, matrix.shape[0], block_size), desc="full-event nearest profiles", unit="block"):
        stop = min(start + block_size, matrix.shape[0]); intersection = (matrix[start:stop] @ matrix.T).toarray().astype(np.float32)
        rows = np.arange(start, stop); intersection[np.arange(stop-start), rows] = -1
        cosine_block = intersection / (norm[start:stop, None] * norm[None, :]); cosine_block[:, sizes == 0] = -1
        best = cosine_block.argmax(axis=1); valid = cosine_block[np.arange(stop-start), best] >= 0
        best[~valid] = -1; neighbor_j[start:stop] = best
        cosine[start:stop][valid] = cosine_block[np.arange(stop-start)[valid], best[valid]]
        common = intersection[np.arange(stop-start)[valid], best[valid]]
        jaccard[start:stop][valid] = common / (sizes[rows[valid]] + sizes[best[valid]] - common)
    return pd.DataFrame({"row": np.arange(matrix.shape[0]), "jaccard_neighbor": neighbor_j, "cosine_similarity": cosine, "jaccard_similarity": jaccard})

def entropy(labels: pd.Series) -> float:
    p = labels.value_counts(normalize=True).to_numpy(); return float(-(p * np.log2(p)).sum())

def run(run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root=base.find_root(Path.cwd()); train=pd.read_csv(root/"data"/"raw"/"train.csv"); genes=[c for c in train if c not in (base.CFG.id_col,base.CFG.target_col)]
    assert train[genes].isna().sum().sum()==0; y=train[base.CFG.target_col].astype(str); cache=base.Cache.build(train[genes],genes)
    profile=cache.events.groupby("row").apply(lambda x: canonical_profile(list(x[["gene","event"]].itertuples(index=False,name=None)))).reindex(range(len(train)),fill_value="WT_ONLY") if not cache.events.empty else pd.Series("WT_ONLY",index=range(len(train)))
    groups=pd.DataFrame({"profile":profile,"label":y}); dup=groups.groupby("profile").agg(rows=("label","size"),labels=("label","nunique"),purity=("label",lambda x:x.value_counts(normalize=True).max()),entropy=("label",entropy)).reset_index()
    duplicate_rows=groups.merge(dup[["profile","rows","labels","purity","entropy"]],on="profile"); ambiguous=duplicate_rows[duplicate_rows.labels.gt(1)]
    matrix=event_matrix(cache); nearest=nearest_profile_metrics(matrix); nearest["label"]=y.to_numpy(); nearest["neighbor_label"]=np.where(nearest.jaccard_neighbor.ge(0), y.to_numpy()[np.maximum(nearest.jaccard_neighbor.to_numpy(),0)], "NO_EVENT_NEIGHBOR"); nearest["label_match"]=nearest.label.eq(nearest.neighbor_label)
    class_nn=nearest[nearest.jaccard_neighbor.ge(0)].groupby("label",as_index=False).agg(nearest_label_match=("label_match","mean"),mean_jaccard=("jaccard_similarity","mean"),mean_cosine=("cosine_similarity","mean"),rows=("label_match","size"))
    burden=np.asarray(cache.mutation.getnnz(axis=1)).ravel(); low=pd.DataFrame({"label":y,"group":np.where(burden==0,"WT_only",np.where(burden<=2,"1_to_2_mutated_genes","3plus_mutated_genes"))}).groupby(["group","label"],as_index=False).size()
    summary=pd.DataFrame([{"experiment_id":"exp-input-audit-01","rows":len(train),"unique_full_event_profiles":int(dup.shape[0]),"duplicate_profile_rows":int(duplicate_rows.rows.gt(1).sum()),"ambiguous_duplicate_rows":int(ambiguous.shape[0]),"ambiguous_duplicate_rate":float(len(ambiguous)/len(train)),"mean_duplicate_purity":float(dup.loc[dup.rows.gt(1),"purity"].mean()),"mean_duplicate_entropy":float(dup.loc[dup.rows.gt(1),"entropy"].mean()),"nearest_label_match_rate":float(nearest.loc[nearest.jaccard_neighbor.ge(0),"label_match"].mean()),"wt_only_rows":int((burden==0).sum()),"leakage_check":True,"nan_as_mutation_count":0}])
    out=root/"experiments"/"gs"/"notebooks"/"exp_model"/"result"; out.mkdir(parents=True,exist_ok=True); summary.to_csv(out/f"{run_id}_summary.csv",index=False); dup.to_csv(out/f"{run_id}_duplicate_profiles.csv",index=False); nearest.to_csv(out/f"{run_id}_nearest.csv",index=False); class_nn.to_csv(out/f"{run_id}_class_nearest.csv",index=False); low.to_csv(out/f"{run_id}_low_mutation_class.csv",index=False)
    return summary,dup,class_nn

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--run-id",default="exp-input-audit-01");a=p.parse_args(); print(run(a.run_id)[0].to_json(orient="records",indent=2))
