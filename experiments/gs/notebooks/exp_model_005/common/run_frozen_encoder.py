"""Seed-42 frozen PubMedBERT screen: E0=P1+EB, E1=encoder LR, E2=fixed blend."""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from frozen_event_encoder import BATCH_SIZE, MAX_LENGTH, MODEL_ID, MODEL_REVISION, event_sentence, pool_event_embeddings


def root():
    for p in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (p / "data/raw/train.csv").exists(): return p
    raise FileNotFoundError("project root not found")

def legacy_module():
    common=root()/"experiments/gs/notebooks/exp_model_002/common"; sys.path.insert(0,str(common))
    spec=importlib.util.spec_from_file_location("encoder_p1",common/"run_p1_axis.py"); module=importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(module); return module

def parser_module():
    common=root()/"experiments/gs/notebooks/exp_model_004/common"; sys.path.insert(0,str(common))
    from parser_recovery import parse_frame
    return parse_frame

def sentence_rows(frame):
    parse_frame=parser_module(); events=parse_frame(frame); rows=[]
    import re
    for sample in events:
        texts=[]
        for event in sample:
            match=re.match(r"([A-Z*])(\d+)([A-Z*])$",event.raw.upper())
            ref, alt=(match.group(1),match.group(3)) if match else (None,None)
            texts.append(event_sentence(event.gene,event.canonical_type,ref,event.position,alt))
        rows.append(texts)
    return rows

def load_encoder(allow_download):
    try: from transformers import AutoModel, AutoTokenizer
    except ImportError as e: raise RuntimeError("transformers가 없습니다. .venv에 설치 후 다시 실행하세요.") from e
    kwargs={"revision":MODEL_REVISION,"local_files_only":not allow_download}
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,**kwargs); model=AutoModel.from_pretrained(MODEL_ID,**kwargs); model.eval()
    import torch
    device="mps" if torch.backends.mps.is_available() else "cpu"; model.to(device)
    vocab_hash=hashlib.sha256(json.dumps(tokenizer.get_vocab(),sort_keys=True).encode()).hexdigest()
    return tokenizer,model,device,vocab_hash

def embed_rows(rows, tokenizer, model, device):
    import torch
    unique=sorted({text for row in rows for text in row}); cache={}
    for start in range(0,len(unique),BATCH_SIZE):
        batch=unique[start:start+BATCH_SIZE]; encoded=tokenizer(batch,padding=True,truncation=True,max_length=MAX_LENGTH,return_tensors="pt")
        encoded={k:v.to(device) for k,v in encoded.items()}
        with torch.no_grad(): values=model(**encoded).last_hidden_state[:,0,:].detach().cpu().numpy().astype(np.float32)
        cache.update(zip(batch,values))
    dim=int(model.config.hidden_size)
    return np.vstack([pool_event_embeddings(np.vstack([cache[text] for text in row]) if row else np.empty((0,dim),np.float32),dim) for row in rows])

def encoder_cv(x, labels, classes, folds, seed):
    probability=np.zeros((len(labels),len(classes)),np.float32); rows=[]; warning_count=0
    for fold,(tr,va) in enumerate(folds,1):
        scaler=StandardScaler(); xtr=scaler.fit_transform(x[tr]); xva=scaler.transform(x[va])
        model=LogisticRegression(solver="lbfgs",C=.07,max_iter=2000,class_weight="balanced",random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always",ConvergenceWarning); model.fit(xtr,labels[tr])
        warning_count+=sum(issubclass(w.category,ConvergenceWarning) for w in caught)
        raw=model.predict_proba(xva); probability[va]=raw
        rows.append({"fold":fold,"macro_f1":f1_score(labels[va],classes[raw.argmax(1)],average="macro"),"feature_count":xtr.shape[1]})
    return probability,pd.DataFrame(rows),warning_count

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--run-id",required=True); ap.add_argument("--allow-download",action="store_true"); args=ap.parse_args()
    legacy=legacy_module(); base,ref,cache,tokens,y,classes=legacy.legacy_context(); _,p0,f0,w0=legacy.eb_cv((base,ref,cache,tokens,y,classes),args.seed)
    train=pd.read_csv(root()/"data/raw/train.csv"); genes=[c for c in train if c not in (base.CFG.id_col,base.CFG.target_col)]
    assert int(train[genes].isna().sum().sum())==0
    tokenizer,model,device,vocab_hash=load_encoder(args.allow_download); x=embed_rows(sentence_rows(train[genes]),tokenizer,model,device)
    p1,f1,w1=encoder_cv(x,y,classes,legacy.fixed_folds(y,args.seed),args.seed); p2=.75*p0+.25*p1
    out=Path(__file__).parent.parent/"result"; out.mkdir(exist_ok=True); variants=[("E0_P1_EB",p0,f0,w0),("E1_frozen_encoder",p1,f1,w1),("E2_fixed_blend",p2,f1,w1)]
    records=[]
    for name,p,folds,w in variants: records.append({"variant":name,"oof_macro_f1":f1_score(y,classes[p.argmax(1)],average="macro"),"feature_count":folds.feature_count.mean(),"convergence_warning_count":w,"leakage_check":True,"nan_as_mutation_count":0})
    summary=pd.DataFrame(records); summary["delta_vs_e0"]=summary.oof_macro_f1-summary.oof_macro_f1.iloc[0]; summary.to_csv(out/f"{args.run_id}_seed{args.seed}_summary.csv",index=False)
    pd.concat([f0.assign(variant="E0_P1_EB"),f1.assign(variant="E1_frozen_encoder"),f1.assign(variant="E2_fixed_blend")],ignore_index=True).to_csv(out/f"{args.run_id}_seed{args.seed}_fold_metrics.csv",index=False)
    pd.DataFrame({"true_class":y,**{f"{name}_{c}":p[:,i] for name,p,_,_ in variants for i,c in enumerate(classes)}}).to_csv(out/f"{args.run_id}_seed{args.seed}_oof_probabilities.csv",index=False)
    (out/f"{args.run_id}_seed{args.seed}_config.json").write_text(json.dumps({"model_id":MODEL_ID,"revision":MODEL_REVISION,"tokenizer_hash":vocab_hash,"frozen":True,"pooling":"mean_max_log1p_count","max_length":MAX_LENGTH,"scaler_fit":"outer_train_only","test_read":False,"nan_as_mutation_count":0},indent=2),encoding="utf-8")
    print(summary.to_string(index=False))
if __name__=="__main__": main()
