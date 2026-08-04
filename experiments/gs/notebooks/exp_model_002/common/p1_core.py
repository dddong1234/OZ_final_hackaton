"""독립 P1 공통 전처리: train-only supervised enrichment, test 미열람."""
from __future__ import annotations
import json, re, warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

AA = tuple('ACDEFGHIKLMNPQRSTVWY')
EVENT_TYPES = ('MISSENSE','NONSENSE','FRAMESHIFT','INFRAME_DEL','INFRAME_INS','SPLICE','OTHER')
EXACT = ('BRAF__V600E','IDH1__R132H','PIK3CA__H1047R','PIK3CA__E545K')
_POS = re.compile(r'(\d+)')
_MISSENSE = re.compile(r'^([A-Z\*])(\d+)([A-Z\*])$')

def project_root() -> Path:
    here = Path(__file__).resolve()
    for p in (here, *here.parents):
        if (p/'data/raw/train.csv').exists(): return p
    raise FileNotFoundError('data/raw/train.csv를 찾지 못했습니다.')

def is_mutation(v) -> bool:
    return isinstance(v, str) and v.strip() not in ('', 'WT', 'wt', 'NAN', 'nan')

def parse_event(v) -> dict:
    if not is_mutation(v): return {'type':'NONE','ref':None,'alt':None,'pos':None,'raw':None}
    s = v.strip().upper()
    if 'FS' in s: typ='FRAMESHIFT'
    elif 'DELINS' in s or ('INS' in s and 'FS' not in s): typ='INFRAME_INS'
    elif 'DEL' in s: typ='INFRAME_DEL'
    elif 'SPLICE' in s: typ='SPLICE'
    elif '*' in s or 'TER' in s or 'X' in s: typ='NONSENSE'
    elif _MISSENSE.match(s): typ='MISSENSE'
    else: typ='OTHER'
    m=_MISSENSE.match(s); pos=_POS.search(s)
    return {'type':typ,'ref':m.group(1) if m else None,'alt':m.group(3) if m else None,
            'pos':int(pos.group(1)) if pos else None,'raw':s}

def event_tokens(frame: pd.DataFrame) -> list[set[str]]:
    genes=list(frame.columns); out=[]
    for row in frame.itertuples(index=False, name=None):
        toks=set()
        for g,v in zip(genes,row):
            e=parse_event(v)
            if e['type']!='NONE': toks.add(f'{g}__{e["type"]}')
        out.append(toks)
    return out

@dataclass
class Cache:
    genes: list[str]; mutation: csr_matrix; exact: np.ndarray; apair: np.ndarray
    burden: np.ndarray; type_count: np.ndarray; topology: np.ndarray; tokens: list[set[str]]; recurrent_tokens: list[set[str]]
    nan_as_mutation_count: int

def build_cache(raw: pd.DataFrame) -> Cache:
    genes=list(raw.columns); n=len(raw); pair_index={a+b:i for i,(a,b) in enumerate((a,b) for a in AA for b in AA if a!=b)}
    rows=[]; cols=[]; exact=np.zeros((n,len(EXACT)),np.float32); apair=np.zeros((n,len(pair_index)),np.float32)
    burden=np.zeros((n,3),np.float32); type_count=np.zeros((n,len(EVENT_TYPES)),np.float32); topology=np.zeros((n,8),np.float32); toks=[]; nan_mut=0
    values=raw.to_numpy(object); recurrent=[]
    for i,row in enumerate(tqdm(values, desc='P1 row-local mutation cache')):
        event_genes=set(); event_types=[]; positions=[]; row_toks=set(); row_recurrent=set()
        for j,v in enumerate(row):
            e=parse_event(v)
            if e['type']=='NONE':
                if pd.isna(v): nan_mut += 0
                continue
            rows.append(i); cols.append(j); g=genes[j]; event_genes.add(g); event_types.append(e['type']); row_toks.add(f'{g}__{e["type"]}')
            if e['pos'] is not None: positions.append(e['pos'])
            type_count[i,EVENT_TYPES.index(e['type'])]+=1
            if e['type']=='MISSENSE': row_recurrent.add(f'{g}__{e["raw"]}')
            key=f'{g}__{e["raw"]}'
            if key in EXACT: exact[i,EXACT.index(key)]=1
            if e['ref'] in AA and e['alt'] in AA and e['ref']!=e['alt']: apair[i,pair_index[e['ref']+e['alt']]]+=1
        cnt=len(event_types); burden[i]=[cnt,len(event_genes),len(set(event_types))]
        topology[i]=[cnt,len(event_genes),len(set(event_types)), max(type_count[i]) if cnt else 0,
                     (max(type_count[i])/cnt if cnt else 0), np.mean(positions) if positions else 0,
                     np.std(positions) if positions else 0, len(positions)]
        toks.append(row_toks)
        recurrent.append(row_recurrent)
    mat=csr_matrix((np.ones(len(rows),np.float32),(rows,cols)),shape=(n,len(genes)),dtype=np.float32)
    return Cache(genes,mat,exact,apair,np.log1p(burden),np.log1p(type_count),np.log1p(topology),toks,recurrent,nan_mut)

def fit_log_odds(tokens: list[set[str]], y: np.ndarray, classes: np.ndarray, *, min_support=10, alpha=1., shrinkage=20., clip=4., empirical_bayes=False) -> dict:
    n=len(y); index={c:i for i,c in enumerate(classes)}; class_n=np.bincount([index[v] for v in y],minlength=len(classes)).astype(float)
    counts={}; totals={}
    for ts,label in zip(tokens,y):
        ci=index[label]
        for t in ts:
            if t not in counts: counts[t]=np.zeros(len(classes),float); totals[t]=0
            counts[t][ci]+=1; totals[t]+=1
    weights={}
    for t,k in totals.items():
        if not empirical_bayes and k<min_support: continue
        cc=counts[t]
        if empirical_bayes:
            p0=(k+alpha)/(n+2*alpha); pc=(cc+shrinkage*p0)/(class_n+shrinkage); notc=((k-cc)+shrinkage*p0)/((n-class_n)+shrinkage)
            w=np.log(np.clip(pc,1e-6,1-1e-6)/(1-np.clip(pc,1e-6,1-1e-6)))-np.log(np.clip(notc,1e-6,1-1e-6)/(1-np.clip(notc,1e-6,1-1e-6)))
        else:
            pc=(cc+alpha)/(class_n+2*alpha); notc=((k-cc)+alpha)/((n-class_n)+2*alpha)
            w=np.log(pc/notc)*(k/(k+shrinkage))
        weights[t]=np.clip(w,-clip,clip).astype(np.float32)
    return weights

def apply_log_odds(tokens: list[set[str]], weights: dict, classes: np.ndarray) -> np.ndarray:
    out=np.zeros((len(tokens),len(classes)),np.float32)
    for i,ts in enumerate(tokens):
        active=[weights[t] for t in ts if t in weights]
        if active: out[i]=np.sum(active,axis=0)/np.sqrt(len(active))
    return out

def recurrent_matrix(cache: Cache, fit_idx: np.ndarray, out_idx: np.ndarray, threshold:int=5) -> csr_matrix:
    """R block: outer fold-train에서 5회 이상인 missense exact event만 binary로 유지."""
    count={}
    for i in fit_idx:
        for t in cache.recurrent_tokens[i]: count[t]=count.get(t,0)+1
    selected=[t for t,n in sorted(count.items()) if n>=threshold]
    vocab={t:j for j,t in enumerate(selected)}
    rows=[]; cols=[]
    for r,i in enumerate(out_idx):
        for t in cache.recurrent_tokens[i]:
            if t in vocab: rows.append(r); cols.append(vocab[t])
    return csr_matrix((np.ones(len(rows),np.float32),(rows,cols)),shape=(len(out_idx),len(vocab)),dtype=np.float32)

def structured(cache: Cache, idx: np.ndarray, recurrent_fit_idx: np.ndarray | None=None) -> csr_matrix:
    dense=np.hstack([cache.exact[idx],cache.apair[idx],cache.burden[idx],cache.type_count[idx],cache.topology[idx]]).astype(np.float32)
    parts=[cache.mutation[idx],csr_matrix(dense)]
    if recurrent_fit_idx is not None: parts.append(recurrent_matrix(cache,recurrent_fit_idx,idx))
    return hstack(parts,format='csr')

def p1_features(cache: Cache, fit_idx: np.ndarray, out_idx: np.ndarray, y: np.ndarray, classes: np.ndarray, *, seed:int, empirical_bayes=False, inner_oof=False):
    if inner_oof:
        score=np.zeros((len(fit_idx),len(classes)),np.float32); splitter=StratifiedKFold(5,shuffle=True,random_state=seed)
        for tr,va in splitter.split(fit_idx,y[fit_idx]):
            ti=fit_idx[tr]; vi=fit_idx[va]; w=fit_log_odds([cache.tokens[i] for i in ti],y[ti],classes,empirical_bayes=empirical_bayes)
            score[va]=apply_log_odds([cache.tokens[i] for i in vi],w,classes)
        return hstack([structured(cache,fit_idx,fit_idx),csr_matrix(score)],format='csr')
    w=fit_log_odds([cache.tokens[i] for i in fit_idx],y[fit_idx],classes,empirical_bayes=empirical_bayes)
    score=apply_log_odds([cache.tokens[i] for i in out_idx],w,classes)
    return hstack([structured(cache,out_idx,fit_idx),csr_matrix(score)],format='csr')

def fit_lr(x,y,seed,ovr=False):
    if ovr:
        from sklearn.multiclass import OneVsRestClassifier
        model=OneVsRestClassifier(LogisticRegression(solver='lbfgs',C=.07,max_iter=2000,class_weight='balanced',random_state=seed))
    else: model=LogisticRegression(solver='lbfgs',C=.07,max_iter=2000,class_weight='balanced',random_state=seed)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always',ConvergenceWarning); model.fit(x,y)
    return model, sum(isinstance(w.message,ConvergenceWarning) for w in ws)

def normalize_proba(p):
    p=np.asarray(p,float); return p/np.clip(p.sum(1,keepdims=True),1e-12,None)

def fixed_folds(y,seed): return list(StratifiedKFold(5,shuffle=True,random_state=seed).split(np.zeros(len(y)),y))

def save_json(path:Path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
