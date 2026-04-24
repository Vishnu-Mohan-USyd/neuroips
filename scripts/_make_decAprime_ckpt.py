"""Task #1 Part B helper — build a ckpt with R1+R2 network state + Dec A' decoder head.

Outputs `/tmp/r1r2_ckpt_decAprime.pt` which is bit-identical to the original R1+R2
checkpoint except:
  - ckpt['loss_heads']['orientation_decoder'] = Dec A' state_dict
  - ckpt['decoder_state']                      = Dec A' state_dict

Both downstream loaders (load_decoder_a in cross_decoder_eval.py; the branch in
r1r2_paradigm_readout.py at :739-745) read from these fields. Patching both is
belt-and-braces. The network itself is untouched, so feeding this ckpt to any
existing pipeline produces R1+R2 network behaviour with Dec A' as the readout.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

_REPO = "/mnt/c/Users/User/codingproj/freshstart"

# Task #2: make src/ importable + register MechanismType so legacy ckpts
# (a1/b1/c1/e1) unpickle cleanly. Previously hard-coded for R1+R2 only.
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from src.config import MechanismType  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-ckpt",
                    default=os.path.join(_REPO, "results/simple_dual/emergent_seed42/checkpoint.pt"))
    ap.add_argument("--dec-a-prime",
                    default=os.path.join(_REPO, "checkpoints/decoder_a_prime.pt"))
    ap.add_argument("--out", default="/tmp/r1r2_ckpt_decAprime.pt")
    args = ap.parse_args()

    print(f"[load] src ckpt: {args.src_ckpt}", flush=True)
    ckpt = torch.load(args.src_ckpt, map_location="cpu", weights_only=False)

    print(f"[load] Dec A' : {args.dec_a_prime}", flush=True)
    dap = torch.load(args.dec_a_prime, map_location="cpu", weights_only=False)
    dap_sd = dap["state_dict"]

    # Sanity check — shapes match
    assert "weight" in dap_sd and dap_sd["weight"].shape == (36, 36), \
        f"Dec A' weight shape mismatch: {dap_sd['weight'].shape}"
    assert "bias" in dap_sd and dap_sd["bias"].shape == (36,), \
        f"Dec A' bias shape mismatch: {dap_sd['bias'].shape}"

    # Patch both places the downstream pipelines read from.
    if "loss_heads" not in ckpt or not isinstance(ckpt["loss_heads"], dict):
        ckpt["loss_heads"] = {}
    old_weight_mean = None
    if "orientation_decoder" in ckpt["loss_heads"]:
        old_w = ckpt["loss_heads"]["orientation_decoder"]["weight"]
        old_weight_mean = float(old_w.mean().item())
    ckpt["loss_heads"]["orientation_decoder"] = {k: v.clone() for k, v in dap_sd.items()}
    ckpt["decoder_state"] = {k: v.clone() for k, v in dap_sd.items()}

    new_weight_mean = float(dap_sd["weight"].mean().item())
    print(
        f"[patch] loss_heads.orientation_decoder swapped "
        f"(old_mean={old_weight_mean}  new_mean={new_weight_mean})",
        flush=True,
    )

    # Add provenance tag so any reader can tell this is a patched ckpt.
    ckpt["_patched_with_decoder_a_prime"] = {
        "src_ckpt": args.src_ckpt,
        "dec_a_prime_path": args.dec_a_prime,
        "dec_a_prime_seed": dap.get("seed"),
        "dec_a_prime_n_steps": dap.get("n_steps"),
        "dec_a_prime_final_val_acc": dap.get("final_val_acc"),
    }

    print(f"[save] writing patched ckpt -> {args.out}", flush=True)
    torch.save(ckpt, args.out)
    print(f"[done] size={os.path.getsize(args.out):,} bytes", flush=True)


if __name__ == "__main__":
    main()
