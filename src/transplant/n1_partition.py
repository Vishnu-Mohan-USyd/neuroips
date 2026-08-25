#!/usr/bin/env python3
"""N1 — G1 (T0-analog partition proof at s=0.04) + G2 (pretrain equality).

G2: per seed, the two regime dirs' common_pretrain_final.pt state_dicts must be
bitwise identical (every key torch.equal + equal key sets).
G1: per regime x seed (8 arms), the pretrain->arm diff must touch ONLY the 7
CELL/FB/GAINS tensors; every other key bitwise identical. Records the delta-norm
table (input to DESIGN 4.1), config equality, k/g per arm and pretrain k (G5 /
4.3 references). Mirrors t0_partition_diff.py conventions (float64 norms).
CPU only, read-only. STOP on any G1/G2 failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncommon as C  # noqa: E402

COMPONENT_OF = {k: comp for comp, keys in C.COMP_TENSORS.items() for k in keys}


def diff_entry(pre_t, arm_t):
    d = (arm_t.to(torch.float64) - pre_t.to(torch.float64))
    return {"shape": list(arm_t.shape),
            "max_abs_diff": float(d.abs().max()) if d.numel() else 0.0,
            "fro_diff": float(d.norm()),
            "fro_pre": float(pre_t.to(torch.float64).norm()),
            "changed": bool(not torch.equal(pre_t, arm_t))}


def main() -> None:
    out = {"G2_pretrain_equality": {}, "seeds": {}, "verdict": None}
    fails, unassigned_changed = [], set()

    for seed in C.SEEDS:
        pres = {a: torch.load(C.pretrain_path(seed, a), map_location="cpu")
                for a in C.ARMS}
        sd0, sd5 = (pres[a]["state_dict"] for a in C.ARMS)
        g2 = (set(sd0) == set(sd5)
              and all(torch.equal(sd0[k], sd5[k]) for k in sd0))
        out["G2_pretrain_equality"][str(seed)] = bool(g2)
        if not g2:
            fails.append(f"G2_seed{seed}")
        pre = pres["alpha0.0"]
        sd_pre = pre["state_dict"]
        entry = {"k_pretrain": C.k_from_raw(sd_pre["circ_raw"]),
                 "g_pretrain": C.g_from_raw(sd_pre["circ_raw"]),
                 "arms": {}}
        for arm in C.ARMS:
            ck = torch.load(C.arm_path(seed, arm), map_location="cpu")
            sd = ck["state_dict"]
            a = {"state_dict_key_sets_equal": set(sd_pre) == set(sd),
                 "k": C.k_from_raw(sd["circ_raw"]),
                 "g": C.g_from_raw(sd["circ_raw"]),
                 "config_equal": ck["tuned_net_config"] == pre["tuned_net_config"],
                 "config_diffs": {k: (pre["tuned_net_config"].get(k), v)
                                  for k, v in ck["tuned_net_config"].items()
                                  if pre["tuned_net_config"].get(k) != v},
                 "surround_config": {
                     "pred_inhib_strength": ck["tuned_net_config"]["pred_inhib_strength"],
                     "pred_inhib_sigma_channels":
                         ck["tuned_net_config"]["pred_inhib_sigma_channels"]},
                 "freeze_local_comp": (pre.get("freeze_local_comp"),
                                       ck.get("freeze_local_comp")),
                 "tensors": {}}
            if not a["state_dict_key_sets_equal"]:
                fails.append(f"G1_keys_seed{seed}_{arm}")
            for key in sorted(set(sd_pre) & set(sd)):
                e = diff_entry(sd_pre[key], sd[key])
                e["component"] = COMPONENT_OF.get(key, "UNASSIGNED")
                a["tensors"][key] = e
                if e["changed"] and e["component"] == "UNASSIGNED":
                    unassigned_changed.add(key)
                    fails.append(f"G1_partition_seed{seed}_{arm}_{key}")
            entry["arms"][arm] = a
            del ck, sd
        out["seeds"][str(seed)] = entry
        del pres, sd0, sd5, pre, sd_pre

    out["unassigned_changed_tensors"] = sorted(unassigned_changed)
    out["verdict"] = {"fails": fails, "PASS": not fails,
                      "partition": ("EXHAUSTIVE" if not unassigned_changed else
                                    "NOT EXHAUSTIVE: " + ",".join(sorted(unassigned_changed)))}
    (C.ROOT / "n1_partition.json").write_text(json.dumps(out, indent=1, sort_keys=True))

    for s in C.SEEDS:
        e = out["seeds"][str(s)]
        print(f"seed {s}: G2={out['G2_pretrain_equality'][str(s)]} "
              f"k pre {e['k_pretrain']:+.4f} -> "
              f"a0.0 {e['arms']['alpha0.0']['k']:+.4f}, "
              f"a0.5 {e['arms']['alpha0.5']['k']:+.4f}; "
              f"config_equal {e['arms']['alpha0.0']['config_equal']}"
              f"/{e['arms']['alpha0.5']['config_equal']}")
        for arm in C.ARMS:
            for key in ("gru.weight_ih", "gru.weight_hh", "W_fb.weight", "circ_raw"):
                t = e["arms"][arm]["tensors"][key]
                rel = (t["fro_diff"] / t["fro_pre"]) if t["fro_pre"] > 0 else float("inf")
                print(f"   {arm} {key:<16} fro_diff {t['fro_diff']:8.3f} rel {rel:.4f}")
    print(out["verdict"]["partition"])
    if fails:
        raise SystemExit(f"N1 GATE FAIL: {fails} — STOP")
    C.heartbeat("N1 PASS: G2 pretrain equality 4/4; G1 partition EXHAUSTIVE 8/8 arms "
                "(diff confined to CELL/FB/GAINS); delta-norm table recorded")


if __name__ == "__main__":
    main()
