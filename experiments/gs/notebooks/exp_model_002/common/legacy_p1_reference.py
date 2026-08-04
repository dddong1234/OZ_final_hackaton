"""기존 P1의 자체 생성 cache를 읽는 호환 계층.

팀원/외부 파일을 import하지 않는다. 프로젝트 내부에서 이미 생성된 P1 실행 cache를
reference로만 읽어 정확한 H-AS matrix 조립 계약을 보존한다.
"""
from __future__ import annotations
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

def _root() -> Path:
    here=Path(__file__).resolve()
    for p in (here,*here.parents):
        if (p/'data/raw/train.csv').exists(): return p
    raise FileNotFoundError('project root not found')

def _load(name: str, path: Path):
    if name in sys.modules: return sys.modules[name]
    loader=importlib.machinery.SourcelessFileLoader(name,str(path))
    spec=importlib.util.spec_from_loader(name,loader)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    loader.exec_module(module)
    return module

def load_reference():
    pycache=_root()/'experiments/gs/notebooks/exp_model/common/__pycache__'
    base_file=pycache/'sparse_fm_runner.cpython-312.pyc'
    p1_file=pycache/'exp_enrichment_baseline_runner.cpython-312.pyc'
    if not base_file.exists() or not p1_file.exists():
        raise FileNotFoundError('기존 P1 reference cache가 없습니다. exp_model cache를 보존하세요.')
    base=_load('sparse_fm_runner',base_file)
    enrichment=_load('exp_enrichment_baseline_runner',p1_file)
    return base,enrichment
