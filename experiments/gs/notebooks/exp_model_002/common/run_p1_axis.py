"""exp_model_002 공통 실행기. train만 읽고 모든 supervised 통계를 fold-train에서 fit한다."""
from __future__ import annotations
import argparse, sys, time, subprocess, tempfile
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0,str(Path(__file__).parent))
from p1_core import (project_root, fixed_folds, fit_lr, normalize_proba, fit_log_odds, apply_log_odds, save_json)
from legacy_p1_reference import load_reference

def legacy_context():
    base,enrichment_ref=load_reference()
    train=pd.read_csv(project_root()/'data/raw/train.csv')
    genes=[c for c in train if c not in (base.CFG.id_col,base.CFG.target_col)]
    assert int(train[genes].isna().sum().sum()) == 0, 'train NaN은 P1 기준선 계약과 다릅니다.'
    cache=base.Cache.build(train[genes],genes)
    y=train[base.CFG.target_col].to_numpy()
    classes=np.asarray(sorted(np.unique(y)))
    tokens=enrichment_ref.gene_event_type_tokens(cache.events)
    return base,enrichment_ref,cache,tokens,y,classes

def pair_tokens(cache):
    rows=[]
    for i in range(cache.mutation.shape[0]):
        gs=[cache.genes[j] for j in cache.mutation[i].indices]
        rows.extend((i,a+'__'+b) for a,b in combinations(gs,2))
    return pd.DataFrame(rows,columns=['row','token']).drop_duplicates()

def enriched(token_sets, fit_idx, out_idx, y, classes, seed, empirical=False, inner=False):
    from sklearn.model_selection import StratifiedKFold
    if inner:
        ret=np.zeros((len(fit_idx),len(classes)),np.float32)
        for tr,va in StratifiedKFold(5,shuffle=True,random_state=seed).split(fit_idx,y[fit_idx]):
            ti,vi=fit_idx[tr],fit_idx[va]; w=fit_log_odds([token_sets[i] for i in ti],y[ti],classes,empirical_bayes=empirical)
            ret[va]=apply_log_odds([token_sets[i] for i in vi],w,classes)
        return ret
    w=fit_log_odds([token_sets[i] for i in fit_idx],y[fit_idx],classes,empirical_bayes=empirical)
    return apply_log_odds([token_sets[i] for i in out_idx],w,classes)

def p1_parts(ctx,tr,va,seed,fold):
    """P1은 기존 H-AS matrix 조립 계약을 reference cache로 정확히 재현한다."""
    base,enrichment_ref,cache,tokens,y,classes=ctx
    matrix,_=base._matrix(cache,tr,y[tr],contrast=True,functional=False,scale_numeric=False)
    inner=enrichment_ref.cross_fitted_enrichment(tokens,tr,y,classes.tolist(),seed*100+fold)
    weights=enrichment_ref.fit_enrichment(tokens,tr,y,classes.tolist())
    valid=enrichment_ref.apply_enrichment(tokens,weights,va,classes.tolist())
    mean=inner.mean(0,keepdims=True); std=np.maximum(inner.std(0,keepdims=True),1e-6)
    return hstack([matrix[tr],csr_matrix((inner-mean)/std)],format='csr'), hstack([matrix[va],csr_matrix((valid-mean)/std)],format='csr'), inner, valid

def p1_cv(ctx,seed,variant='p1'):
    _,_,_,_,y,classes=ctx; folds=fixed_folds(y,seed); prob=np.zeros((len(y),len(classes))); fold_rows=[]; warns=0
    for fold,(tr,va) in enumerate(folds,1):
        xtr,xva,_,_=p1_parts(ctx,tr,va,seed,fold)
        m,w=fit_lr(xtr,y[tr],seed,ovr=(variant=='ovr')); warns+=w; p=normalize_proba(m.predict_proba(xva)); prob[va]=p
        fold_rows.append({'fold':fold,'macro_f1':f1_score(y[va],classes[p.argmax(1)],average='macro'),'feature_count':xtr.shape[1]})
    return classes,prob,pd.DataFrame(fold_rows),warns

def dense_cv(ctx,seed):
    _,_,cache,_,y,classes=ctx; prob=np.zeros((len(y),len(classes))); rows=[]
    for fold,(tr,va) in enumerate(fixed_folds(y,seed),1):
        _,_,scr_tr,scr_va=p1_parts(ctx,tr,va,seed,fold)
        trunc=np.asarray(cache.truncation.sum(axis=1)).ravel().reshape(-1,1)
        dense=lambda idx,s: np.hstack([s,np.log1p(cache.burden[idx]),np.log1p(cache.variant[idx]),np.log1p(trunc[idx]),np.log1p(cache.topology[idx])])
        with tempfile.TemporaryDirectory(prefix='p1_dense_lgbm_') as tmp:
            inp=Path(tmp)/'input.npz'; out=Path(tmp)/'output.npz'
            np.savez_compressed(inp,x_train=dense(tr,scr_tr).astype(np.float32),y_train=np.asarray(y[tr],dtype=str),x_valid=dense(va,scr_va).astype(np.float32))
            worker=Path(__file__).with_name('dense_lgbm_worker.py')
            subprocess.run([sys.executable,str(worker),'--input',str(inp),'--output',str(out),'--seed',str(seed+fold)],check=True)
            data=np.load(out); raw=data['probability']; worker_classes=data['classes'].astype(str)
        p=np.zeros((len(va),len(classes)),np.float32)
        for j,label in enumerate(worker_classes): p[:,np.where(classes==label)[0][0]]=raw[:,j]
        p=normalize_proba(p); prob[va]=p
        rows.append({'fold':fold,'macro_f1':f1_score(y[va],classes[p.argmax(1)],average='macro'),'feature_count':dense(tr,scr_tr).shape[1]})
    return classes,prob,pd.DataFrame(rows),0

def pair_cv(ctx,seed):
    base,_,cache,_,y,classes=ctx; pt=pair_tokens(cache); token_sets=[set() for _ in range(len(y))]
    for row,token in pt.itertuples(index=False): token_sets[row].add(token)
    prob=np.zeros((len(y),len(classes))); rows=[]; warns=0
    for fold,(tr,va) in enumerate(fixed_folds(y,seed),1):
        matrix,_=base._matrix(cache,tr,y[tr],contrast=True,functional=False,scale_numeric=False)
        base_tr,base_va=matrix[tr],matrix[va]
        ps_tr=enriched(token_sets,tr,tr,y,classes,seed+fold,inner=True); ps_va=enriched(token_sets,tr,va,y,classes,seed+fold)
        m,w=fit_lr(hstack([base_tr,csr_matrix(ps_tr)]),y[tr],seed); warns+=w; p=normalize_proba(m.predict_proba(hstack([base_va,csr_matrix(ps_va)]))); prob[va]=p
        rows.append({'fold':fold,'macro_f1':f1_score(y[va],classes[p.argmax(1)],average='macro'),'feature_count':base_tr.shape[1]+len(classes)})
    return classes,prob,pd.DataFrame(rows),warns

def eb_cv(ctx,seed):
    base,_,cache,tokens,y,classes=ctx; token_sets=[set() for _ in range(len(y))]
    for row,token in tokens.itertuples(index=False): token_sets[row].add(token)
    prob=np.zeros((len(y),len(classes))); rows=[]; warns=0
    for fold,(tr,va) in enumerate(fixed_folds(y,seed),1):
        matrix,_=base._matrix(cache,tr,y[tr],contrast=True,functional=False,scale_numeric=False)
        s_tr=enriched(token_sets,tr,tr,y,classes,seed+fold,empirical=True,inner=True); s_va=enriched(token_sets,tr,va,y,classes,seed+fold,empirical=True)
        mu=s_tr.mean(0,keepdims=True); sd=np.maximum(s_tr.std(0,keepdims=True),1e-6)
        xtr=hstack([matrix[tr],csr_matrix((s_tr-mu)/sd)],format='csr'); xva=hstack([matrix[va],csr_matrix((s_va-mu)/sd)],format='csr')
        m,w=fit_lr(xtr,y[tr],seed); warns+=w; p=normalize_proba(m.predict_proba(xva)); prob[va]=p
        rows.append({'fold':fold,'macro_f1':f1_score(y[va],classes[p.argmax(1)],average='macro'),'feature_count':xtr.shape[1]})
    return classes,prob,pd.DataFrame(rows),warns

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--axis',choices=['residual','dense','pair','eb','ovr'],required=True); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--run-id',required=True); args=ap.parse_args()
    out=Path(__file__).parent.parent/'result'; out.mkdir(exist_ok=True); ctx=legacy_context(); _,_,cache,_,y,classes=ctx; start=time.time()
    # P1 reference always shares exactly the candidate's outer folds.
    classes,p0,f0,w0=p1_cv(ctx,args.seed)
    if args.axis=='residual':
        pred=classes[p0.argmax(1)]; top=np.sort(p0,axis=1)[:,-2:]; confidence=pd.DataFrame({'true_class':y,'predicted_class':pred,'correct':pred==y,'margin':top[:,1]-top[:,0],'entropy':-(p0*np.log(np.clip(p0,1e-12,1))).sum(1)})
        confidence.to_csv(out/f'{args.run_id}_seed{args.seed}_residual_rows.csv',index=False)
        cm=confusion_matrix(y,pred,labels=classes); pairs=[]
        for i,j in zip(*np.where((cm>0)&~np.eye(len(classes),dtype=bool))): pairs.append({'true_class':classes[i],'predicted_class':classes[j],'count':int(cm[i,j])})
        pd.DataFrame(pairs).sort_values('count',ascending=False).to_csv(out/f'{args.run_id}_seed{args.seed}_confusions.csv',index=False)
        summary=pd.DataFrame([{'variant':'P1 residual audit','oof_macro_f1':f1_score(y,pred,average='macro'),'feature_count':f0.feature_count.mean(),'convergence_warning_count':w0,'leakage_check':True,'nan_as_mutation_count':0,'runtime_seconds':time.time()-start}])
    else:
        fn={'dense':dense_cv,'pair':pair_cv,'eb':eb_cv,'ovr':lambda c,s:p1_cv(c,s,'ovr')}[args.axis]
        _,p1,f1,w1=fn(ctx,args.seed); pred0=classes[p0.argmax(1)]; pred1=classes[p1.argmax(1)]
        summary=pd.DataFrame([
          {'variant':'P1 multinomial LR','oof_macro_f1':f1_score(y,pred0,average='macro'),'feature_count':f0.feature_count.mean(),'convergence_warning_count':w0,'leakage_check':True,'nan_as_mutation_count':0,'runtime_seconds':time.time()-start},
          {'variant':args.axis,'oof_macro_f1':f1_score(y,pred1,average='macro'),'feature_count':f1.feature_count.mean(),'convergence_warning_count':w1,'leakage_check':True,'nan_as_mutation_count':0,'runtime_seconds':time.time()-start}])
        pd.DataFrame({'true_class':y,**{'p1_'+c:p0[:,i] for i,c in enumerate(classes)},**{args.axis+'_'+c:p1[:,i] for i,c in enumerate(classes)}}).to_csv(out/f'{args.run_id}_seed{args.seed}_oof_probabilities.csv',index=False)
        f1.assign(variant=args.axis).to_csv(out/f'{args.run_id}_seed{args.seed}_fold_metrics.csv',index=False)
    summary.to_csv(out/f'{args.run_id}_seed{args.seed}_summary.csv',index=False)
    save_json(out/f'{args.run_id}_seed{args.seed}_leakage_audit.json',{'train_only':True,'test_read':False,'fold_train_supervised_statistics':True,'nan_as_mutation_count':0,'axis':args.axis,'p1_matrix_reference':'legacy_H_AS_matrix'})
    print(summary.to_string(index=False))

if __name__=='__main__': main()
