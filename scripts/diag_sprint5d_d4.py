"""Sprint 5d D4: route impulse-response (3 seeds).

Wrapper around `expectation_snn/scripts/diag_route_impulse.py` to run
all three seeds {42, 43, 44} per pre-reg §3 D4 and save outputs to
data/diag_sprint5d/ with the canonical D4_seed{N}.npz filename.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRIVER = ROOT / "expectation_snn" / "scripts" / "diag_route_impulse.py"
SRC_DIR = ROOT / "expectation_snn" / "data"
OUT_DIR = ROOT / "data" / "diag_sprint5d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44]
PY = "/home/vysoforlife/miniconda3/envs/expectation_snn/bin/python"


def main() -> int:
    t0 = time.time()
    for seed in SEEDS:
        out_path = OUT_DIR / f"D4_seed{seed}.npz"
        if out_path.exists():
            print(f"[D4 seed={seed}] SKIP (already saved)")
            continue
        t1 = time.time()
        print(f"[D4 seed={seed}] running ...")
        cp = subprocess.run(
            [PY, str(DRIVER), "--seed", str(seed),
             "--checkpoint-seed", "42", "--quiet"],
            cwd=str(ROOT), check=False, capture_output=True, text=True,
        )
        if cp.returncode != 0:
            print(f"[D4 seed={seed}] FAILED rc={cp.returncode}")
            print("STDERR (last 500 chars):", cp.stderr[-500:])
            return 1
        src = SRC_DIR / f"diag_route_impulse_seed{seed}.npz"
        if not src.exists():
            print(f"[D4 seed={seed}] expected output missing: {src}")
            return 2
        shutil.copy2(src, out_path)
        print(f"[D4 seed={seed}] done in {time.time()-t1:.1f}s -> {out_path}")
    print(f"[D4] ALL DONE in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
