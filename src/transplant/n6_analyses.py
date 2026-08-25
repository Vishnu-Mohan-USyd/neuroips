#!/usr/bin/env python3
"""N6 — deeper strategy analyses (DESIGN 4.1-4.3). Post-processing only.

4.1 per-set trained-vs-pretrain deltas: ||Delta||_F and relative (float64), SVD
    of Delta_hh (64x64) and Delta_fb (36x64): sigma_1..10 + top-5 energy e5.
4.2 FB alignment geometry: per-row cosine (median, frac>0.9); whole-matrix
    normalized inner product; cross-set coupling E_proj = ||Delta_fb V5||^2_F /
    ||Delta_fb||^2_F with V5 = top-5 right singular vectors of that regime's
    Delta_hh (analytic random null 5/64 = 0.078125); functional alignment:
    PCA of final-step h over the standard battery on the pure arm (TTT),
    a_k = ||W_fb u_k||_2 for trained vs pretrain FB on the SAME trained h-PCs.
4.3 gain/k configuration vs the no-surround originals (read from the frozen
    original study's t0_partition_diff.json).
GPU used only for the h-PCA forward; everything else CPU float64.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncommon as C  # noqa: E402
from ncommon import assay, tuned  # noqa: E402

ORIG_T0 = Path("/home/vishnu/neuroips_analysis/transplant_20260818/t0_partition_diff.json")
TENSORS = [k for keys in C.COMP_TENSORS.values() for k in keys]


def svd_stats(d64: torch.Tensor) -> dict:
    s = torch.linalg.svdvals(d64)
    tot = float((s ** 2).sum())
    return {"sigma_1_to_10": [float(x) for x in s[:10]],
            "e5_top5_energy": float((s[:5] ** 2).sum()) / tot if tot > 0 else 0.0}


@torch.no_grad()
def main() -> None:
    C.mem_gate()
    device = assay.choose_device("cuda:0")
    out = {"deltas": {}, "fb_geometry": {}, "gains_table": {},
           "original_no_surround_reference": {}}

    orig = json.loads(ORIG_T0.read_text())
    for s in C.SEEDS:
        e = orig["seeds"][str(s)]
        out["original_no_surround_reference"][str(s)] = {
            "k_pretrain": e["k_pretrain"],
            "k": {a: e["arms"][a]["k"] for a in C.ARMS},
            "g": {a: e["arms"][a]["g"] for a in C.ARMS},
            "fro_diff": {a: {t: e["arms"][a]["tensors"][t]["fro_diff"]
                             for t in TENSORS} for a in C.ARMS}}

    for seed in C.SEEDS:
        pre_ck = torch.load(C.pretrain_path(seed), map_location="cpu")
        sd_pre = pre_ck["state_dict"]
        g_pre = C.g_from_raw(sd_pre["circ_raw"])
        k_pre = C.k_from_raw(sd_pre["circ_raw"])
        for arm in C.ARMS:
            ck = torch.load(C.arm_path(seed, arm), map_location="cpu")
            sd = ck["state_dict"]
            key = f"seed{seed}_{arm}"

            # ---------------------------------------------- 4.1 deltas + SVD
            dl = {}
            for t in TENSORS:
                d = sd[t].to(torch.float64) - sd_pre[t].to(torch.float64)
                fp = float(sd_pre[t].to(torch.float64).norm())
                dl[t] = {"fro_diff": float(d.norm()),
                         "fro_pre": fp,
                         "rel": float(d.norm()) / fp if fp > 0 else None}
            d_hh = sd["gru.weight_hh"].to(torch.float64) - \
                sd_pre["gru.weight_hh"].to(torch.float64)
            d_fb = sd["W_fb.weight"].to(torch.float64) - \
                sd_pre["W_fb.weight"].to(torch.float64)
            dl["SVD_delta_hh"] = svd_stats(d_hh)
            dl["SVD_delta_fb"] = svd_stats(d_fb)
            out["deltas"][key] = dl

            # ------------------------------------------- 4.2 FB geometry
            w_t = sd["W_fb.weight"].to(torch.float64)
            w_p = sd_pre["W_fb.weight"].to(torch.float64)
            row_cos = torch.nn.functional.cosine_similarity(w_t, w_p, dim=1)
            whole = float((w_t * w_p).sum() / (w_t.norm() * w_p.norm()))
            _, _, vh = torch.linalg.svd(d_hh)
            v5 = vh[:5].T  # 64x5 top-5 right singular vectors of Delta_hh
            e_proj = float((d_fb @ v5).norm() ** 2 / d_fb.norm() ** 2)
            # functional alignment: h-PCA on the pure arm (TTT = the arm ckpt)
            net, ck_l = assay.load_arm(C.arm_path(seed, arm), device)
            mode = tuned.resolve_feedback_mode(
                bool(ck_l.get("center_feedback", False)), ck_l.get("feedback_mode"))
            theta_a, _, _ = assay.matched_pairs(device)
            rec = C.instrumented_unroll(net, theta_a, mode, device)
            h_fin = rec["h_seq"][:, -1, :].to(torch.float64).cpu()
            h_c = h_fin - h_fin.mean(dim=0, keepdim=True)
            _, sv, vh_h = torch.linalg.svd(h_c, full_matrices=False)
            pcs = vh_h[:5]  # 5x64
            a_of = {
                "trained": [float((w_t @ pcs[i]).norm()) for i in range(5)],
                "pretrain": [float((w_p @ pcs[i]).norm()) for i in range(5)]}
            out["fb_geometry"][key] = {
                "row_cos_median": float(row_cos.median()),
                "row_cos_frac_gt_0p9": float((row_cos > 0.9).to(torch.float64).mean()),
                "whole_matrix_inner": whole,
                "E_proj_delta_fb_on_delta_hh_V5": e_proj,
                "E_proj_random_null": 5.0 / 64.0,
                "h_pca_top5_var_frac": [float(x) for x in
                                        (sv[:5] ** 2 / (sv ** 2).sum())],
                "a_k_Wfb_on_trained_hPCs": a_of}
            C.release(net, rec, h_fin, h_c)

            # ------------------------------------------- 4.3 gains table
            g_arm = C.g_from_raw(sd["circ_raw"])
            out["gains_table"][key] = {
                "g": g_arm, "k": C.k_from_raw(sd["circ_raw"]),
                "som_margin": g_arm[1] - g_arm[2] * g_arm[0],
                "g_pretrain": g_pre, "k_pretrain": k_pre,
                "k_original_no_surround":
                    out["original_no_surround_reference"][str(seed)]["k"][arm]}
            del ck, sd
        del pre_ck, sd_pre

    (C.ROOT / "n6_analyses.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    for key in sorted(out["fb_geometry"]):
        fg, dl = out["fb_geometry"][key], out["deltas"][key]
        print(f"{key}: row_cos_med {fg['row_cos_median']:+.3f} "
              f"whole {fg['whole_matrix_inner']:+.3f} "
              f"E_proj {fg['E_proj_delta_fb_on_delta_hh_V5']:.3f} "
              f"e5(hh) {dl['SVD_delta_hh']['e5_top5_energy']:.3f} "
              f"|d_fb| {dl['W_fb.weight']['fro_diff']:.2f}")
    C.heartbeat("N6 DONE: deltas/SVD, FB geometry (row-cos, E_proj, h-PCA "
                "functional alignment), gains table written")


if __name__ == "__main__":
    main()
