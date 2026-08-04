"""LightGBM worker는 PyTorch/P1 reference를 import하지 않는 독립 프로세스 계약이다."""
import subprocess, sys, tempfile
from pathlib import Path
import numpy as np

def test_clean_worker_roundtrip():
    worker=Path(__file__).parent/'dense_lgbm_worker.py'
    with tempfile.TemporaryDirectory() as d:
        d=Path(d); rng=np.random.default_rng(42)
        y=np.asarray(['A']*20+['B']*20+['C']*20,dtype=str)
        np.savez_compressed(d/'input.npz',x_train=rng.normal(size=(60,12)).astype('float32'),y_train=y,x_valid=rng.normal(size=(6,12)).astype('float32'))
        subprocess.run([sys.executable,str(worker),'--input',str(d/'input.npz'),'--output',str(d/'out.npz'),'--seed','42'],check=True)
        p=np.load(d/'out.npz')['probability']
        assert p.shape==(6,3) and np.isfinite(p).all() and np.allclose(p.sum(1),1)

if __name__=='__main__':
    test_clean_worker_roundtrip(); print('dense worker test passed')
