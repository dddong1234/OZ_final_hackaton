import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]; exp=root/"exp"
def c(kind,text): return {"cell_type":kind,"metadata":{},"source":[line+"\n" for line in text.strip().splitlines()],**({"execution_count":None,"outputs":[]} if kind=="code" else {})}
cells=[
c("markdown","""# Frozen biomedical event encoder — seed 42 screen

고정 PubMedBERT를 제공된 mutation 문자열에만 적용합니다. tokenizer·encoder는 학습하지 않으며, test는 읽지 않습니다. E0=P1+EB, E1=encoder LR, E2=0.75/0.25 고정 blend입니다."""),
c("code",'''from pathlib import Path
import importlib.util, subprocess, sys
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
ROOT=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/"experiments/gs/notebooks/exp_model_005/common/run_frozen_encoder.py").exists())
RUNNER=ROOT/"experiments/gs/notebooks/exp_model_005/common/run_frozen_encoder.py"
RESULT=ROOT/"experiments/gs/notebooks/exp_model_005/result"
print({"torch":importlib.util.find_spec("torch") is not None,"transformers":importlib.util.find_spec("transformers") is not None})
print("모델 ID: microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
print("모델 revision: 2839b4fc440a3c41dc2b716fb14d530c33c8c1ff")'''),
c("code",'''SEED=42
RUN_EXPERIMENT=False
ALLOW_MODEL_DOWNLOAD=False
RUN_ID="exp-frozen-biomedical-encoder-01"

# transformers가 없다면 notebook 밖의 .venv에서 한 번만 설치하세요:
# /Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m pip install transformers
# 설치 후에도 ALLOW_MODEL_DOWNLOAD=True를 명시해야 고정 pretrained weight를 다운로드합니다.
if RUN_EXPERIMENT:
    if not ALLOW_MODEL_DOWNLOAD:
        raise RuntimeError("고정 pretrained model download를 명시적으로 허용하려면 ALLOW_MODEL_DOWNLOAD=True로 변경하세요.")
    command=[sys.executable,str(RUNNER),"--seed",str(SEED),"--run-id",RUN_ID,"--allow-download"]
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1); tail=[]
    for line in tqdm(process.stdout,desc="frozen encoder runner",unit="line"):
        print(line,end=""); tail=(tail+[line])[-120:]
    if process.wait(): raise RuntimeError("runner failed:\\n"+"".join(tail))
else: print("RUN_EXPERIMENT=False: dependency/configuration check only")'''),
c("code",'''path=RESULT/f"{RUN_ID}_seed{SEED}_summary.csv"
if path.exists():
    summary=pd.read_csv(path); folds=pd.read_csv(RESULT/f"{RUN_ID}_seed{SEED}_fold_metrics.csv")
    display(summary); display(folds)
    assert summary.leakage_check.all() and summary.nan_as_mutation_count.eq(0).all()
    summary.set_index("variant")["oof_macro_f1"].plot.bar(figsize=(8,4),title="Frozen encoder screen"); plt.tight_layout(); plt.show()
else: print("실행 후 결과 CSV가 생성됩니다.")''')]
payload={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python (.venv)","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
(exp/"exp-frozen-biomedical-encoder-01.ipynb").write_text(json.dumps(payload,ensure_ascii=False,indent=1),encoding="utf-8")
