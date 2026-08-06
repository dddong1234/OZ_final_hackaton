#!/usr/bin/env bash
# Exact-event EB 3-seed 재현 러너.
#
# 항상 --output-dir 를 이 폴더의 out/ 으로 고정한다.
# (스크립트 기본 출력 경로가 experiments/gs/notebooks/submission/ 이고
#  기본 파일명이 경수님 제출 파일과 동일해서, 인자 없이 돌리면 덮어쓴다.)
#
# 사용법:
#   bash run_repro.sh smoke     # 몇 초. test.csv 를 읽지 않는다.
#   bash run_repro.sh seed42    # 단일 seed. 시간 측정용.
#   bash run_repro.sh 3seed     # 최종 3-seed 재현.
#   bash run_repro.sh compare   # 경수님 제출 CSV 와 일치율 비교.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
OUT="$HERE/out"
SCRIPT="$HERE/reproduce_exact_event_eb_3seed.py"
PY="${PY:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

mkdir -p "$OUT"
echo "[run_repro] python  = $PY"
echo "[run_repro] root    = $ROOT"
echo "[run_repro] output  = $OUT"

case "${1:-smoke}" in
  smoke)
    "$PY" "$SCRIPT" --root "$ROOT" --smoke
    ;;
  seed42)
    "$PY" - "$SCRIPT" "$ROOT" "$OUT" <<'PYEOF'
import importlib.util, pathlib, sys, time
script, root, out = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
spec = importlib.util.spec_from_file_location("repro", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.ROOT_OVERRIDE = root
module.OUTPUT_DIR_OVERRIDE = out
started = time.time()
module.run(output_name="ij_repro_seed42.csv")
elapsed = time.time() - started
print(f"[run_repro] seed42 elapsed_sec = {elapsed:.1f}  (3-seed 예상 약 {elapsed * 3 / 60:.1f}분)")
PYEOF
    ;;
  3seed)
    time "$PY" "$SCRIPT" \
      --root "$ROOT" \
      --output-dir "$OUT" \
      --output-name ij_repro_3seed.csv
    ;;
  compare)
    REF="${2:-$ROOT/experiments/gs/notebooks/submission/submission_h0_exact_event_eb_seed42_777_2024_bagged.csv}"
    "$PY" "$HERE/compare_submissions.py" "$OUT/ij_repro_3seed.csv" "$REF"
    ;;
  *)
    echo "usage: bash run_repro.sh [smoke|seed42|3seed|compare [reference.csv]]" >&2
    exit 2
    ;;
esac
