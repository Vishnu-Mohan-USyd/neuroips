#!/usr/bin/env python3
"""N5 — s->0 inference counterfactual on hybrids (DESIGN 2.4, flank-A4 style).

Registered list: {TTT, TPT, PPT, PTP, PPP} x 2 regimes x seed 8, plus every
seed-8 cell whose rho_flank >= 0.25 in its factorial (bounded <= 8 more; PPP
is one shared net — measured once, entered in both tables, disclosed).

SELECTION CORRECTION (disclosed, first run superseded): the rho_flank >= 0.25
rule is applied with the DESIGN 3.3 machinery — rho on a coordinate exists only
when |TTT-PPP| clears the 0.05 flank floor. Seed-8 alpha0.0 flank denominator
= 0.01764 < 0.05 => alpha0.0 rho_flank UNREADABLE; per 3.3 the flank question
there "is answered descriptively (raw flank + the s->0 counterfactual)", so
ALL non-registered alpha0.0 cells get the s->0 re-assay as the descriptive set
(bucket extras_floored_descriptive). The rule-selected extras come from the
readable factorial only (alpha0.5), cap 8. The first run selected extras by
raw rho numbers off the floored denominator (8 alpha0.0 cells, missing TTP and
the two readable alpha0.5 candidates); this rerun supersedes it.

Counterfactual: rebuild from the checkpoint's own config with
pred_inhib_strength=0 (sigma inert at s=0, flank-validator-proven), same
state_dict, same assay. Evidence, never a bar. Device cuda:0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncommon as C  # noqa: E402
from ncommon import assay  # noqa: E402

REGISTERED = ("TTT", "TPT", "PPT", "PTP", "PPP")
PROFILE_KEYS = ("H", "center_ratio", "flank_ratio", "M_auc_ratio",
                "continuation_mean_rate")


def main() -> None:
    C.mem_gate()
    device = assay.choose_device("cuda:0")
    n4 = json.loads((C.ROOT / "n4_assay.json").read_text())["per_seed"]["8"]

    FLANK_FLOOR = 0.05
    extras, descriptive, floored_arms = [], [], []
    for arm in C.ARMS:
        f_ppp = n4[arm]["PPP"]["profile"]["flank_ratio"]
        f_ttt = n4[arm]["TTT"]["profile"]["flank_ratio"]
        den = f_ttt - f_ppp
        if abs(den) < FLANK_FLOOR:
            floored_arms.append({"arm": arm, "flank_den": den,
                                 "floor": FLANK_FLOOR})
            descriptive += [(arm, cid) for cid in sorted(n4[arm])
                            if cid not in REGISTERED]
            continue
        for cid, cell in sorted(n4[arm].items()):
            if cid in REGISTERED:
                continue
            rho = (cell["profile"]["flank_ratio"] - f_ppp) / den
            if rho >= 0.25:
                extras.append((arm, cid, rho))
    extras = extras[:8]

    out = {"device": str(device), "registered": {}, "extras": {},
           "extras_floored_descriptive": {},
           "extra_selection_rule": "seed-8 rho_flank >= 0.25 on READABLE "
                                   "factorials (|TTT-PPP| >= 0.05), cap 8; "
                                   "floored factorials get the full "
                                   "descriptive s->0 set per DESIGN 3.3",
           "floored_factorials": floored_arms,
           "ppp_note": "PPP is one shared net; measured once, entered in both "
                       "factorial tables"}
    todo = [("registered", arm, cid) for arm in C.ARMS for cid in REGISTERED
            if not (cid == "PPP" and arm == "alpha0.5")]
    todo += [("extras", arm, cid) for arm, cid, _ in extras]
    todo += [("extras_floored_descriptive", arm, cid)
             for arm, cid in descriptive]
    n_done = 0
    for bucket, arm, cid in todo:
        C.mem_gate()
        p = C.cell_ckpt(8, arm, cid)
        official = n4[arm][cid]["profile"]
        cf = C.measure_path_s0(p, device)
        ent = {"official": {k: official[k] for k in PROFILE_KEYS},
               "s0_counterfactual": {k: cf[k] for k in PROFILE_KEYS},
               "delta_s_minus_s0": {k: official[k] - cf[k] for k in PROFILE_KEYS}}
        key = f"{arm}_{cid}"
        out[bucket][key] = ent
        if cid == "PPP":
            out[bucket]["alpha0.5_PPP"] = dict(ent, shared_with="alpha0.0_PPP")
        n_done += 1
        if n_done % 10 == 0:
            C.heartbeat(f"N5: {n_done} s->0 re-assays done (latest {key})")
        C.release()

    out["extras_selected"] = [{"arm": a, "cell": c, "rho_flank_s8": r}
                              for a, c, r in extras]
    (C.ROOT / "n5_s0.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({"n_reassays": n_done,
                      "extras_selected": out["extras_selected"]}, indent=1))
    C.heartbeat(f"N5 DONE: {n_done} s->0 re-assays "
                f"({len(out['extras_selected'])} extras by rho_flank rule)")


if __name__ == "__main__":
    main()
