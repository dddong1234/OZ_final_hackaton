import importlib.util
from pathlib import Path
import sys

path=Path(__file__).with_name("exp_deep_sets_runner.py"); spec=importlib.util.spec_from_file_location("deep_sets",path); module=importlib.util.module_from_spec(spec); assert spec and spec.loader; sys.modules[spec.name]=module; spec.loader.exec_module(module)

def test_event_encoder_excludes_wt_and_keeps_real_events():
    assert module.encode_event("TP53","R175H","MISSENSE")[0] is not None
    assert module.encode_event("TP53","WT","MISSENSE") is None

def test_runner_never_references_test_csv():
    assert '"test.csv"' not in path.read_text(encoding="utf-8")

if __name__=="__main__":
    test_event_encoder_excludes_wt_and_keeps_real_events(); test_runner_never_references_test_csv(); print("Deep Sets contracts passed")
