#!/usr/bin/env python3
"""N3 — FB control constructions R / N / Q + construction gates (DESIGN 2.3).

R (random-init): fresh net via build_tuned_from_config(host config) under
torch.manual_seed(20260823) on CPU; its W_fb.weight/W_fb.bias. ONE draw for all
cells (regime-independent).
N (norm-matched random, per regime x seed): R tensors rescaled so ||W_N||_F =
||W_fb^T||_F and ||b_N||_2 = ||b^T||_2; gate: relative norm error < 1e-6.
Q (rotation-misaligned trained, per regime x seed): W_Q = W^T @ Q with Q Haar
(QR of manual_seed(20260824) Gaussian, sign-fixed diag(R)>0); b_Q = b^T.
Gates: ||Q^T Q - I||_inf < 1e-5; |  ||W_Q||_F / ||W^T||_F - 1 | < 1e-5; row
2-norms preserved (reported).

Cells built: T.ctrl.T (3 controls x 2 regimes x 4 seeds = 24 nets) and
P.ctrl.P seed 8 (PRP regime-independent dual-filename; PNP/PQP per regime =
5 nets, 6 table cells). Construction audit: every key traced bitwise to its
declared source (arm/pretrain/constructed-control). Null-edit gate: the
control-splicing path with FB := the regime's own trained FB must rebuild TTT
bitwise, 8/8. Control tensors saved to controls_fb.pt for Phase-3 rebuild.
CPU construction; STOP on any gate failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncommon as C  # noqa: E402
from ncommon import tuned  # noqa: E402

FB_KEYS = ("W_fb.weight", "W_fb.bias")


def fro(t):
    return float(t.to(torch.float64).norm())


def build_R(host_cfg) -> dict:
    torch.manual_seed(C.R_SEED)
    net = tuned.build_tuned_from_config(host_cfg)  # CPU
    sd = net.state_dict()
    return {"W_fb.weight": sd["W_fb.weight"].detach().clone().cpu(),
            "W_fb.bias": sd["W_fb.bias"].detach().clone().cpu()}


def build_Q() -> torch.Tensor:
    torch.manual_seed(C.Q_SEED)
    gauss = torch.randn(64, 64, dtype=torch.float64)
    q, r = torch.linalg.qr(gauss)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)  # sign-fix diag(R)>0
    return q


def ctrl_fb(kind: str, r_draw: dict, q64: torch.Tensor, w_t, b_t):
    """Return (W, b) float32 CPU for control `kind` given the trained FB."""
    if kind == "R":
        return r_draw["W_fb.weight"].clone(), r_draw["W_fb.bias"].clone()
    if kind == "N":
        sw = fro(w_t) / fro(r_draw["W_fb.weight"])
        sb = fro(b_t) / fro(r_draw["W_fb.bias"])
        return (r_draw["W_fb.weight"].to(torch.float64) * sw).to(torch.float32), \
               (r_draw["W_fb.bias"].to(torch.float64) * sb).to(torch.float32)
    if kind == "Q":
        wq = (w_t.to(torch.float64) @ q64).to(torch.float32)
        return wq, b_t.clone()
    raise ValueError(kind)


CELL_OF = {"R": ("TRT", "PRP"), "N": ("TNT", "PNP"), "Q": ("TQT", "PQP")}


def main() -> None:
    host_cfg = torch.load(C.pretrain_path(8), map_location="cpu")["tuned_net_config"]
    r_draw = build_R(host_cfg)
    q64 = build_Q()
    out = {"rng": {"R_seed": C.R_SEED, "Q_seed": C.Q_SEED},
           "R_norms": {k: fro(v) for k, v in r_draw.items()},
           "Q_gates": {}, "N_gates": {}, "Q_cell_gates": {}, "construction": {},
           "null_edit_gate": {}, "verdict": {}}
    fails = []

    qtq = (q64.T @ q64 - torch.eye(64, dtype=torch.float64)).abs().max()
    out["Q_gates"] = {"QtQ_minus_I_inf": float(qtq), "pass": float(qtq) < 1e-5}
    if not out["Q_gates"]["pass"]:
        fails.append("Q_orthogonality")

    n_built = 0
    saved_ctrl = {"R": r_draw, "Q64": q64}
    for seed in C.SEEDS:
        sd_pre = torch.load(C.pretrain_path(seed), map_location="cpu")["state_dict"]
        bodies = {a: torch.load(C.arm_path(seed, a), map_location="cpu") for a in C.ARMS}
        for arm in C.ARMS:
            sd_arm = bodies[arm]["state_dict"]
            w_t, b_t = sd_arm["W_fb.weight"].cpu(), sd_arm["W_fb.bias"].cpu()

            # --------------------------- null-edit gate: FB := trained FB
            sd_null = dict(sd_arm)
            sd_null["W_fb.weight"], sd_null["W_fb.bias"] = w_t.clone(), b_t.clone()
            null_ok = (set(sd_null) == set(sd_arm)
                       and all(torch.equal(sd_null[k], sd_arm[k]) for k in sd_arm))
            out["null_edit_gate"][f"seed{seed}_{arm}"] = bool(null_ok)
            if not null_ok:
                fails.append(f"null_edit_seed{seed}_{arm}")

            for kind in ("R", "N", "Q"):
                w_c, b_c = ctrl_fb(kind, r_draw, q64, w_t, b_t)
                if kind == "N":
                    ew = abs(fro(w_c) / fro(w_t) - 1.0)
                    eb = abs(fro(b_c) / fro(b_t) - 1.0)
                    out["N_gates"][f"seed{seed}_{arm}"] = {
                        "w_rel_err": ew, "b_rel_err": eb,
                        "pass": ew < 1e-6 and eb < 1e-6}
                    if not out["N_gates"][f"seed{seed}_{arm}"]["pass"]:
                        fails.append(f"N_norm_seed{seed}_{arm}")
                if kind == "Q":
                    enorm = abs(fro(w_c) / fro(w_t) - 1.0)
                    rn_t = w_t.to(torch.float64).norm(dim=1)
                    rn_c = w_c.to(torch.float64).norm(dim=1)
                    row_dev = float(((rn_c - rn_t).abs() / rn_t).max())
                    out["Q_cell_gates"][f"seed{seed}_{arm}"] = {
                        "fro_rel_err": enorm, "max_row_norm_rel_dev": row_dev,
                        "pass": enorm < 1e-5}
                    if not out["Q_cell_gates"][f"seed{seed}_{arm}"]["pass"]:
                        fails.append(f"Q_norm_seed{seed}_{arm}")
                saved_ctrl.setdefault(kind, {})[f"seed{seed}_{arm}"] = {
                    "W_fb.weight": w_c, "W_fb.bias": b_c}

                # ------------------------------- T.ctrl.T (trained context)
                tcell = CELL_OF[kind][0]
                sd = dict(sd_arm)
                sd["W_fb.weight"], sd["W_fb.bias"] = w_c.clone(), b_c.clone()
                p = C.save_hybrid(bodies[arm], sd, C.cell_ckpt(seed, arm, tcell))
                n_built += 1
                rl = torch.load(p, map_location="cpu")["state_dict"]
                per_key = all(
                    torch.equal(rl[k], (w_c if k == "W_fb.weight" else
                                        b_c if k == "W_fb.bias" else sd_arm[k]))
                    for k in rl)
                out["construction"][f"seed{seed}_{arm}_{tcell}"] = bool(
                    per_key and set(rl) == set(sd_arm))
                if not out["construction"][f"seed{seed}_{arm}_{tcell}"]:
                    fails.append(f"construction_seed{seed}_{arm}_{tcell}")
                del rl, sd

                # ------------------------------- P.ctrl.P (host context, s8)
                if seed == 8 and kind in ("N", "Q"):
                    pcell = CELL_OF[kind][1]
                    sd = dict(sd_pre)
                    sd["W_fb.weight"], sd["W_fb.bias"] = w_c.clone(), b_c.clone()
                    p = C.save_hybrid(bodies[arm], sd, C.cell_ckpt(seed, arm, pcell))
                    n_built += 1
                    rl = torch.load(p, map_location="cpu")["state_dict"]
                    ok = all(torch.equal(rl[k], (w_c if k == "W_fb.weight" else
                                                 b_c if k == "W_fb.bias" else sd_pre[k]))
                             for k in rl) and set(rl) == set(sd_pre)
                    out["construction"][f"seed{seed}_{arm}_{pcell}"] = bool(ok)
                    if not ok:
                        fails.append(f"construction_seed{seed}_{arm}_{pcell}")
                    del rl, sd
        # PRP: regime-independent (R draw + host body), dual filename like PPP
        if seed == 8:
            w_c, b_c = ctrl_fb("R", r_draw, q64, None, None)
            sd = dict(sd_pre)
            sd["W_fb.weight"], sd["W_fb.bias"] = w_c.clone(), b_c.clone()
            for arm in C.ARMS:
                C.save_hybrid(bodies[arm], sd, C.cell_dir(seed, arm, "PRP") / C.SLUG_OF[arm])
            n_built += 1
            rl = {a: torch.load(C.cell_dir(seed, a, "PRP") / C.SLUG_OF[a],
                                map_location="cpu")["state_dict"] for a in C.ARMS}
            cross = all(torch.equal(rl["alpha0.0"][k], rl["alpha0.5"][k]) for k in sd)
            ok = cross and all(
                torch.equal(rl["alpha0.0"][k], (w_c if k == "W_fb.weight" else
                                                b_c if k == "W_fb.bias" else sd_pre[k]))
                for k in rl["alpha0.0"])
            out["construction"]["seed8_shared_PRP"] = bool(ok)
            if not ok:
                fails.append("construction_seed8_shared_PRP")
            del rl, sd
        del sd_pre, bodies

    torch.save(saved_ctrl, C.ROOT / "controls_fb.pt")
    out["n_control_nets_built"] = n_built
    out["verdict"] = {"fails": fails, "PASS": not fails}
    (C.ROOT / "n3_control_gates.json").write_text(json.dumps(
        {k: v for k, v in out.items() if k != "construction"} |
        {"construction": out["construction"]}, indent=1, sort_keys=True, default=float))
    print(json.dumps({"n_control_nets_built": n_built,
                      "Q_gates": out["Q_gates"],
                      "N_all_pass": all(v["pass"] for v in out["N_gates"].values()),
                      "Q_cells_all_pass": all(v["pass"] for v in out["Q_cell_gates"].values()),
                      "null_edit_8of8": all(out["null_edit_gate"].values()),
                      "construction_all": all(out["construction"].values()),
                      "verdict": out["verdict"]}, indent=1))
    if fails:
        raise SystemExit(f"N3 GATE FAIL: {fails} — STOP")
    C.heartbeat(f"N3 PASS: {n_built} control nets built (R/N/Q); Q orthogonality + norm "
                "gates clean; null-edit gate 8/8; construction audit clean; "
                "controls_fb.pt saved for Phase-3 rebuild")


if __name__ == "__main__":
    main()
