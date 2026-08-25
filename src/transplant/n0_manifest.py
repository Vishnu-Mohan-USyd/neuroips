#!/usr/bin/env python3
"""N0 — G6 donor manifest (DESIGN 3.6), written BEFORE any build.

sha256 of every file in all 8 donor dirs -> MANIFEST.sha256 + n0_manifest.json.
Gate: the seed-8 pretrain STATE sha (harness state_sha256 recipe) must equal
the recorded 4c5b1a32... in BOTH regime dirs. Re-verified after all
measurement (n7). CPU only, read-only on donors.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncommon as C  # noqa: E402


def main() -> None:
    files = {}
    for seed in C.SEEDS:
        for arm in C.ARMS:
            d = C.donor_dir(seed, arm)
            if not d.is_dir():
                raise SystemExit(f"G6 FAIL: donor dir missing {d}")
            for p in sorted(d.iterdir()):
                if p.is_file():
                    files[str(p)] = C.sha256_file(p)
    lines = [f"{sha}  {path}" for path, sha in sorted(files.items())]
    (C.ROOT / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")

    state_shas = {}
    for arm in C.ARMS:
        ck = torch.load(C.pretrain_path(8, arm), map_location="cpu")
        state_shas[arm] = C.state_sha256(ck["state_dict"])
        del ck
    ok = all(v == C.PRETRAIN_S8_STATE_SHA for v in state_shas.values())
    out = {"n_files": len(files),
           "seed8_pretrain_state_sha256": state_shas,
           "recorded_anchor": C.PRETRAIN_S8_STATE_SHA,
           "G6_anchor_pass": ok,
           "manifest": files}
    (C.ROOT / "n0_manifest.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({k: out[k] for k in
                      ("n_files", "seed8_pretrain_state_sha256", "G6_anchor_pass")},
                     indent=1))
    if not ok:
        raise SystemExit("G6 FAIL: seed-8 pretrain state sha != recorded anchor — STOP")
    C.heartbeat(f"N0/G6 PASS: manifest {len(files)} files; seed-8 pretrain state sha "
                "matches recorded anchor in both regime dirs")


if __name__ == "__main__":
    main()
