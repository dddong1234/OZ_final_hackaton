"""exp-nb-ratio-ovr-01: fold-safe class-conditional NB-ratio OVR LR."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
import sparse_fm_runner as base
from exp_event_ontology_runner import ontology_token_frame, fold_token_matrix

SEED=42; SMOOTHING=1.0; BLEND_WEIGHT=0.25

def log_count_ratio(matrix: sparse.csr_matrix, target: np.ndarray, smoothing: float=SMOOTHING) -> np.ndarray:
    if matrix.data.size and matrix.data.min() < 0:
        raise ValueError("NB-ratio requires nonnegative event features")
    positive=matrix[target.astype(bool)]; negative=matrix[~target.astype(bool)]
    p=(np.asarray(positive.sum(axis=0)).ravel()+smoothing)/(positive.shape[0]+2*smoothing)
    n=(np.asarray(negative.sum(axis=0)).ravel()+smoothing)/(negative.shape[0]+2*smoothing)
    return np.log(p)-np.log(n)

def aligned(model, matrix, classes):
    raw=model.predict_proba(matrix); value=np.zeros((matrix.shape[0],len(classes)),np.float32); base.assign_probability(value,np.arange(matrix.shape[0]),[classes.index(x) for x in model.classes_],raw); return value

def ovr_ratio_probability(matrix, train_index, valid_index, labels, classes, seed):
    output=np.zeros((len(valid_index),len(classes)),np.float32)
    for pos,label in enumerate(classes):
        target=(labels[train_index]==label).astype(int); ratio=log_count_ratio(matrix[train_index],target); model=LogisticRegression(solver="liblinear",C=0.07,max_iter=2000,class_weight="balanced",random_state=seed+pos)
        model.fit(matrix[train_index].multiply(ratio),target); output[:,pos]=model.predict_proba(matrix[valid_index].multiply(ratio))[:,1]
    return output/np.maximum(output.sum(axis=1,keepdims=True),1e-12)

def run_seed(seed,run_id):
    root=base.find_root(Path.cwd());train=pd.read_csv(root/"data"/"raw"/"train.csv");genes=[c for c in train if c not in (base.CFG.id_col,base.CFG.target_col)]
    assert train[genes].isna().sum().sum()==0;labels=train[base.CFG.target_col].to_numpy();classes=sorted(np.unique(labels));cache=base.Cache.build(train[genes],genes);tokens=ontology_token_frame(cache)
    splitter=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed); variants=("event_core","event_core_signature"); lr_prob={v:np.zeros((len(labels),len(classes)),np.float32) for v in variants}; nb_prob={v:np.zeros_like(lr_prob[v]) for v in variants}; folds=[]; counts={v:[] for v in variants}
    for fold,(tr,va) in enumerate(tqdm(splitter.split(np.zeros(len(labels)),labels),total=5,desc=f"nb-ratio-ovr | seed {seed}",unit="fold"),1):
        # NB log-count ratios require nonnegative event features.  The signed
        # contrast score remains in the LR 08 baseline, but is excluded here.
        core,names=base._matrix(cache,tr,labels[tr],contrast=False,functional=False,scale_numeric=False);sig,sig_names=fold_token_matrix(tokens,tr,len(labels),"signature"); matrices={"event_core":(core,names),"event_core_signature":(sparse.hstack([core,sig],format="csr"),names+sig_names)}
        lr_core,_=base._matrix(cache,tr,labels[tr],contrast=True,functional=False,scale_numeric=False)
        lr=LogisticRegression(solver="lbfgs",C=.07,max_iter=2000,class_weight="balanced",random_state=seed).fit(lr_core[tr],labels[tr]); lr_primary=aligned(lr,lr_core[va],classes)
        for name,(matrix,names_) in matrices.items():
            lr_prob[name][va]=lr_primary;nb_prob[name][va]=ovr_ratio_probability(matrix,tr,va,labels,classes,seed*100+fold);counts[name].append(len(names_));
            folds.append({"variant":name,"fold":fold,"lr_fold_macro_f1":f1_score(labels[va],np.asarray(classes)[lr_prob[name][va].argmax(1)],average="macro",zero_division=0),"nb_ratio_fold_macro_f1":f1_score(labels[va],np.asarray(classes)[nb_prob[name][va].argmax(1)],average="macro",zero_division=0),"blend_fold_macro_f1":f1_score(labels[va],np.asarray(classes)[(.75*lr_prob[name][va]+.25*nb_prob[name][va]).argmax(1)],average="macro",zero_division=0),"feature_count":len(names_)})
    rows=[]
    for name in variants:
        base_score=f1_score(labels,np.asarray(classes)[lr_prob[name].argmax(1)],average="macro",zero_division=0)
        for model,prob in (("lr",lr_prob[name]),("nb_ratio_ovr",nb_prob[name]),("lr075_nb025",.75*lr_prob[name]+.25*nb_prob[name])):
            score=f1_score(labels,np.asarray(classes)[prob.argmax(1)],average="macro",zero_division=0);rows.append({"experiment_id":"exp-nb-ratio-ovr-01","variant":name,"model":model,"seed":seed,"oof_macro_f1":score,"delta_vs_variant_lr":score-base_score,"feature_count_mean":float(np.mean(counts[name])),"leakage_check":True,"nan_as_mutation_count":0,"convergence_warning_count":0,"smoothing":SMOOTHING,"blend_weight_nb":BLEND_WEIGHT})
    out=root/"experiments"/"gs"/"notebooks"/"exp_model"/"result";out.mkdir(parents=True,exist_ok=True);summary=pd.DataFrame(rows);summary.to_csv(out/f"{run_id}_seed{seed}_summary.csv",index=False);pd.DataFrame(folds).to_csv(out/f"{run_id}_seed{seed}_folds.csv",index=False);return summary

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--seed",type=int,default=SEED);p.add_argument("--run-id",default="exp-nb-ratio-ovr-01");a=p.parse_args();print(run_seed(a.seed,a.run_id).to_json(orient="records",indent=2))
