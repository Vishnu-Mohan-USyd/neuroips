"""Task #5 helper — build a ckpt with R1+R2 net state + Dec E in the decoder slot.

Pattern mirrors ``scripts/_make_decAprime_ckpt.py``. Replaces
``ckpt['loss_heads']['orientation_decoder']`` and ``ckpt['decoder_state']``
with Dec E weights. Downstream `load_decoder_a` in ``cross_decoder_eval.py``
and the branch in ``r1r2_paradigm_readout.py`` will then read Dec E, so the
pipelines' output ``decA_delta`` values are Δ_E.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

_REPO = "/mnt/c/Users/User/codingproj/freshstart"
sys.path.insert(0, _REPO)

# Force import so the MechanismType enum is available for legacy ckpt unpickling.
from src.config import MechanismType  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-ckpt",
                    default=os.path.join(_REPO, "results/simple_dual/emergent_seed42/checkpoint.pt"))
    ap.add_argument("--dec-e",
                    default=os.path.join(_REPO, "checkpoints/decoder_e.pt"))
    ap.add_argument("--out", default="/tmp/r1r2_ckpt_decE.pt")
    args = ap.parse_args()

    print(f"[load] src ckpt: {args.src_ckpt}", flush=True)
    ckpt = torch.load(args.src_ckpt, map_location="cpu", weights_only=False)

    print(f"[load] Dec E:  {args.dec_e}", flush=True)
    de = torch.load(args.dec_e, map_location="cpu", weights_only=False)
    de_sd = de["state_dict"]
    assert de_sd["weight"].shape == (36, 36), f"Dec E weight shape: {de_sd['weight'].shape}"
    assert de_sd["bias"].shape == (36,), f"Dec E bias shape: {de_sd['bias'].shape}"

    if "loss_heads" not in ckpt or not isinstance(ckpt["loss_heads"], dict):
        ckpt["loss_heads"] = {}
    ckpt["loss_heads"]["orientation_decoder"] = {k: v.clone() for k, v in de_sd.items()}
    ckpt["decoder_state"] = {k: v.clone() for k, v in de_sd.items()}
    ckpt["_patched_with_decoder_e"] = {
        "src_ckpt": args.src_ckpt,
        "dec_e_path": args.dec_e,
        "dec_e_seed": de.get("seed"),
        "dec_e_n_steps": de.get("n_steps"),
        "dec_e_final_val_acc": de.get("final_val_acc"),
    }
    print(f"[patch] loss_heads.orientation_decoder ← Dec E  "
          f"(new weight mean={de_sd['weight'].mean().item():.4e})", flush=True)

    print(f"[save] → {args.out}", flush=True)
    torch.save(ckpt, args.out)
    print(f"[done] size={os.path.getsize(args.out):,} bytes", flush=True)


if __name__ == "__main__":
    main()
