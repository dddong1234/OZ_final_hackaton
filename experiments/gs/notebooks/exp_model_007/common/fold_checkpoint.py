"""Atomic, fold-level checkpoints for long-running OOF experiments."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def experiment_result_dir(runner_path: Path) -> Path:
    """Resolve `<experiment>/result`, never `<experiment>/common/result`."""
    return runner_path.resolve().parent.parent / "result"


def save_checkpoint(path: Path, payload: dict) -> None:
    """Persist OOF arrays and fold metadata together; retain the prior file on interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"oof__{name}": np.asarray(value) for name, value in payload["oof"].items()}
    metadata = {
        "completed_folds": sorted(int(fold) for fold in payload["completed_folds"]),
        "fold_rows": payload["fold_rows"],
        "audit_rows": payload["audit_rows"],
    }
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, metadata_json=np.asarray(json.dumps(metadata)), **arrays)
    os.replace(temporary, path)
    progress_path = path.with_suffix(".progress.json")
    progress_temporary = progress_path.with_suffix(".tmp.json")
    progress_temporary.write_text(json.dumps({"completed_folds": metadata["completed_folds"]}, indent=2), encoding="utf-8")
    os.replace(progress_temporary, progress_path)


def load_checkpoint(path: Path) -> dict | None:
    """Load a complete checkpoint, or return None when no completed fold was saved."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        oof = {key.removeprefix("oof__"): archive[key].copy() for key in archive.files if key.startswith("oof__")}
    return {**metadata, "oof": oof}
