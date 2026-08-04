"""PyTorch를 import하지 않는 clean-process LightGBM worker."""
from __future__ import annotations
import argparse
import numpy as np
from lightgbm import LGBMClassifier

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',required=True); p.add_argument('--seed',type=int,required=True); a=p.parse_args()
    d=np.load(a.input); xtr=d['x_train'].astype(np.float32,copy=False); y=d['y_train']; xva=d['x_valid'].astype(np.float32,copy=False)
    classes=np.unique(y)
    model=LGBMClassifier(objective='multiclass',num_class=len(classes),n_estimators=500,learning_rate=.03,num_leaves=15,max_depth=4,min_child_samples=30,subsample=.8,colsample_bytree=.8,reg_lambda=5.,reg_alpha=.2,class_weight='balanced',random_state=a.seed,n_jobs=-1,verbosity=-1)
    model.fit(xtr,y)
    probability=model.predict_proba(xva).astype(np.float32)
    if not np.isfinite(probability).all(): raise RuntimeError('LightGBM probability is not finite')
    np.savez_compressed(a.output,probability=probability,classes=classes)

if __name__=='__main__': main()
