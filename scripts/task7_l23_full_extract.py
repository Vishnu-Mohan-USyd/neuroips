"""
Task #7: pure data-extraction expansion of Task #4.
Reads existing R1+R2 result JSONs (no GPU runs) and emits, per pathway × bucket:
  - Full 36-ch re-centered L2/3 mean tuning curve (probe window)
  - mean across all 36 channels
  - argmax channel (relative to center_idx=18)
  - 2nd-highest channel (to reveal off-peak bumps for clean-march Unexpected)
  - half-height activity (for shape inspection beyond FWHM)
  - notes when time-course / per-trial std are NOT stored

No interpretation. Read-only. Stdout only — no JSON written.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

RESULTS = Path("/mnt/c/Users/User/codingproj/freshstart/results")
JSONS = {
    "matched_3row_ring": "matched_3row_ring_r1_2.json",
    "matched_hmm_ring_sequence": "matched_hmm_ring_sequence_r1_2.json",
    "matched_hmm_ring_sequence_tightexp": "matched_hmm_ring_sequence_tightexp_r1_2.json",
    "matched_probe_3pass": "matched_probe_3pass_r1_2.json",
    "v2_confidence_dissection": "v2_confidence_dissection_r1_2.json",
}

CENTER = 18  # re-centered probe target channel
N_CH = 36


def fmt(x, w=8, p=4):
    if x is None:
        return "  --  "
    if isinstance(x, (int,)):
        return f"{x:>{w}d}"
    return f"{x:>{w}.{p}f}"


def summarize_ring(ring: list[float]) -> dict:
    """Compute summary stats for a 36-ch tuning curve, re-centered at CENTER=18."""
    assert len(ring) == N_CH, f"expected 36 channels, got {len(ring)}"
    total = sum(ring)
    mean_all = total / N_CH
    argmax = max(range(N_CH), key=lambda i: ring[i])
    sorted_idx = sorted(range(N_CH), key=lambda i: ring[i], reverse=True)
    second_idx = sorted_idx[1]
    peak_value = ring[argmax]
    second_value = ring[second_idx]
    # signed displacement from center, wrapped to [-18, 18]
    displ = argmax - CENTER
    if displ > 18:
        displ -= 36
    elif displ <= -18:
        displ += 36
    second_displ = second_idx - CENTER
    if second_displ > 18:
        second_displ -= 36
    elif second_displ <= -18:
        second_displ += 36
    half_max = peak_value / 2.0
    n_above_half = sum(1 for v in ring if v >= half_max)
    return {
        "argmax_idx": argmax,
        "argmax_displ_from_center": displ,
        "peak": peak_value,
        "second_idx": second_idx,
        "second_displ_from_center": second_displ,
        "second_value": second_value,
        "second_to_peak_ratio": second_value / peak_value if peak_value > 0 else None,
        "mean_across_36ch": mean_all,
        "total_sum": total,
        "n_channels_above_half_peak": n_above_half,
        "min": min(ring),
        "max": peak_value,
    }


def report_3row_ring(d: dict):
    print("=" * 80)
    print("PATHWAY: matched_3row_ring (HMM pred-err split, single combined window)")
    print("=" * 80)
    print(f"checkpoint: {d['checkpoint']}")
    print(f"meta keys present: {sorted(d['meta'].keys())}")
    print("NOTE: Has only `mean_ring` per bucket (combined window). NO mean_pm2/mean_pm1/mean_probe split.")
    print("NOTE: NO time-course stored. NO per-trial std stored.")
    print()
    for bk in ("expected", "unexpected", "omission"):
        b = d["buckets"][bk]
        s = summarize_ring(b["mean_ring"])
        print(f"-- BUCKET: {bk}  (n={b['n']}) --")
        print(f"   reported peak_at_true   : {fmt(b['peak_at_true'])}")
        print(f"   reported total          : {fmt(b['total'])}")
        print(f"   reported FWHM (deg)     : {fmt(b['fwhm_deg'], p=2)}")
        print(f"   reported decoder_acc    : {fmt(b['decoder_acc'])}")
        print(f"   reported mean_pi_pred   : {fmt(b['mean_pi_pred_eff'])}")
        print(f"   --- TASK-7 ADDITIONS (from mean_ring) ---")
        print(f"   argmax channel          : {s['argmax_idx']}  (displ from ch18 = {s['argmax_displ_from_center']:+d})")
        print(f"   peak value (== max)     : {fmt(s['peak'])}")
        print(f"   2nd-highest channel     : {s['second_idx']}  (displ = {s['second_displ_from_center']:+d}, value={fmt(s['second_value'])}, ratio2/1={fmt(s['second_to_peak_ratio'])})")
        print(f"   mean across 36 ch       : {fmt(s['mean_across_36ch'])}")
        print(f"   total (sum 36 ch)       : {fmt(s['total_sum'])}")
        print(f"   #chs >= peak/2          : {s['n_channels_above_half_peak']}")
        print(f"   min channel value       : {fmt(s['min'])}")
        print(f"   per-trial std at peak   :   not stored in JSON")
        print(f"   per-trial std on total  :   not stored in JSON")
        print(f"   time course (ON/ISI)    :   not stored in JSON (readout fixed at [9,11])")
        print(f"   FULL 36-ch mean_ring    :")
        for i in range(0, N_CH, 12):
            row = b["mean_ring"][i : i + 12]
            print("     " + "  ".join(f"{v:.4f}" for v in row))
        print()


def report_hmm_seq(d: dict, label: str):
    print("=" * 80)
    print(f"PATHWAY: {label} (HMM, 3-window storage: pm2 / pm1 / probe)")
    print("=" * 80)
    print(f"checkpoint: {d['checkpoint']}")
    print(f"meta windows: pm2/pm1/probe each one tuning curve per bucket (no time series)")
    if "tight_expected_used" in d:
        print(f"tight_expected_used={d['tight_expected_used']}, exp_pred_err_max={d.get('exp_pred_err_max_used')}, unexp_pred_err_min={d.get('unexp_pred_err_min_used')}")
    print(f"underpowered={d['underpowered']}, min_bucket_n_floor={d['min_bucket_n_floor']}")
    print("NOTE: NO time-course stored. NO per-trial std stored. Only mean across trials per channel.")
    print()
    for bk in ("expected", "unexpected", "omission"):
        b = d["buckets"][bk]
        s_pm2 = summarize_ring(b["mean_pm2"])
        s_pm1 = summarize_ring(b["mean_pm1"])
        s_pr = summarize_ring(b["mean_probe"])
        print(f"-- BUCKET: {bk}  (n={b['n']}, n_cw={b['n_cw']}, n_ccw={b['n_ccw']}) --")
        print(f"   reported peak_at_true_probe : {fmt(b['peak_at_true_probe'])}")
        print(f"   reported total_probe        : {fmt(b['total_probe'])}")
        print(f"   reported FWHM_probe (deg)   : {fmt(b['fwhm_probe_deg'], p=2)}")
        print(f"   reported decoder_acc        : {fmt(b['decoder_acc'])}")
        print(f"   reported mean_pi_pred_eff   : {fmt(b['mean_pi_pred_eff'])}")
        print(f"   reported mean_pred_err      : {fmt(b['mean_pred_err'])}")
        print(f"   --- TASK-7 ADDITIONS ---")
        for win, s, name in [("pm2", s_pm2, "mean_pm2"), ("pm1", s_pm1, "mean_pm1"), ("probe", s_pr, "mean_probe")]:
            print(f"   [{win}] argmax_ch={s['argmax_idx']:>2d} (displ {s['argmax_displ_from_center']:+d}), "
                  f"peak={fmt(s['peak'])}, 2nd_ch={s['second_idx']:>2d} (displ {s['second_displ_from_center']:+d}), "
                  f"2nd_val={fmt(s['second_value'])}, ratio2/1={fmt(s['second_to_peak_ratio'])}, "
                  f"mean36={fmt(s['mean_across_36ch'])}, #>=peak/2={s['n_channels_above_half_peak']}")
        print(f"   per-trial std (any window)  :   not stored in JSON")
        print(f"   time course (ON/ISI window) :   not stored in JSON (probe summary uses readout [9,11])")
        print(f"   FULL 36-ch mean_probe (re-centered, ch18 = probe true_ch) :")
        for i in range(0, N_CH, 12):
            row = b["mean_probe"][i : i + 12]
            print("     " + "  ".join(f"{v:.4f}" for v in row))
        print()


def report_3pass(d: dict):
    print("=" * 80)
    print("PATHWAY: matched_probe_3pass (CLEAN-MARCH 3-pass design)")
    print("=" * 80)
    print(f"checkpoint: {d['checkpoint']}")
    print(f"underpowered={d['underpowered']}, n_qualifying={d['n_qualifying']}")
    print(f"widening_cascade: {d['widening_cascade']}")
    print(f"context_identical_pm2_across_rows={d['context_identical_pm2_across_rows']}, "
          f"context_identical_pm1_across_rows={d['context_identical_pm1_across_rows']}")
    print(f"flip_ccw={d['flip_ccw']}")
    print()
    print("NOTE: pm2 and pm1 windows are IDENTICAL across exp/unexp/omission by construction "
          "(passes A,B,C share the same pre-probe state). Only mean_probe diverges.")
    print("NOTE: NO time-course stored. NO per-trial std stored.")
    print()
    for bk in ("expected", "unexpected", "omission"):
        b = d["buckets"][bk]
        s_pm2 = summarize_ring(b["mean_pm2"])
        s_pm1 = summarize_ring(b["mean_pm1"])
        s_pr = summarize_ring(b["mean_probe"])
        print(f"-- BUCKET: {bk}  (n={b['n']}, n_cw={b['n_cw']}, n_ccw={b['n_ccw']}) --")
        print(f"   reported peak_at_true_probe : {fmt(b['peak_at_true_probe'])}")
        print(f"   reported total_probe        : {fmt(b['total_probe'])}")
        print(f"   reported FWHM_probe (deg)   : {fmt(b['fwhm_probe_deg'], p=2)}")
        print(f"   reported decoder_acc        : {fmt(b['decoder_acc'])}")
        print(f"   reported mean_pi_pred_eff   : {fmt(b['mean_pi_pred_eff'])}")
        print(f"   reported mean_pred_err_A    : {fmt(b['mean_pred_err_A'])}")
        print(f"   --- TASK-7 ADDITIONS (probe-window 36-ch tuning curve) ---")
        for win, s, name in [("pm2", s_pm2, "mean_pm2"), ("pm1", s_pm1, "mean_pm1"), ("probe", s_pr, "mean_probe")]:
            print(f"   [{win}] argmax_ch={s['argmax_idx']:>2d} (displ {s['argmax_displ_from_center']:+d}), "
                  f"peak={fmt(s['peak'])}, 2nd_ch={s['second_idx']:>2d} (displ {s['second_displ_from_center']:+d}), "
                  f"2nd_val={fmt(s['second_value'])}, ratio2/1={fmt(s['second_to_peak_ratio'])}, "
                  f"mean36={fmt(s['mean_across_36ch'])}, #>=peak/2={s['n_channels_above_half_peak']}")
        print(f"   per-trial std (any window)  :   not stored in JSON")
        print(f"   time course (ON/ISI window) :   not stored in JSON (probe summary uses readout [9,11])")
        print(f"   FULL 36-ch mean_probe (re-centered, ch18 = probe true_ch) :")
        for i in range(0, N_CH, 12):
            row = b["mean_probe"][i : i + 12]
            print("     " + "  ".join(f"{v:.4f}" for v in row))
        # Specifically asked: where is the mean_probe argmax for unexpected?
        if bk == "unexpected":
            print(f"   *** non-peak-anchored bump audit (Task-7 §5) ***")
            print(f"       argmax of unexpected mean_probe is at ch={s_pr['argmax_idx']} "
                  f"(NOT ch18=probe true_ch). Displacement from probe true_ch = "
                  f"{s_pr['argmax_displ_from_center']:+d} channels = {s_pr['argmax_displ_from_center']*5}°.")
            print(f"       value at probe true_ch (ch18) = {b['mean_probe'][CENTER]:.4f}")
            print(f"       value at argmax (ch{s_pr['argmax_idx']})   = {b['mean_probe'][s_pr['argmax_idx']]:.4f}")
            # Look for second-bump cluster on the opposite side of the ring
            print(f"       full vector channels 30-35 + 0-5 (ring-wrap region):")
            wrap = [b['mean_probe'][i % N_CH] for i in list(range(30, 36)) + list(range(0, 6))]
            print("         " + "  ".join(f"ch{(30+i)%36}={v:.4f}" for i, v in enumerate(wrap)))
        print()


def report_dissection(d: dict):
    print("=" * 80)
    print("PATHWAY: v2_confidence_dissection (Tests 1, 2, 3)")
    print("=" * 80)
    print(f"checkpoint: {d['checkpoint']}")
    print("NOTE: This script stores ONLY summary stats (peak_at_true, total, dec_acc).")
    print("NOTE: NO 36-ch tuning curve stored per bucket. NO time-course. NO per-trial std.")
    print("NOTE: Test 2 = quartile-binned HMM exp vs unexp. Test 3 = pi-matched HMM pairs.")
    print()
    print("--- TEST 1: pi_pred_eff distribution per group (no L2/3 vectors) ---")
    for grp_key, grp in d["test1"]["groups"].items():
        print(f"   {grp_key:25s} n={grp['n']:>5d}  pi mean={fmt(grp['mean'])}  median={fmt(grp['median'])}  "
              f"std={fmt(grp['std'])}  p10={fmt(grp['p10'])}  p25={fmt(grp['p25'])}  p75={fmt(grp['p75'])}  p90={fmt(grp['p90'])}")
    print(f"   ks_hmm_exp_vs_unexp: D={d['test1']['ks_hmm_exp_vs_unexp']['D']:.4f}, p={d['test1']['ks_hmm_exp_vs_unexp']['p']:.2e}")
    print(f"   ks_hmm_exp_vs_march: D={d['test1']['ks_hmm_exp_vs_march']['D']:.4f}, p={d['test1']['ks_hmm_exp_vs_march']['p']:.2e}")
    print(f"   note: {d['test1']['note']}")
    print()
    print("--- TEST 2: quartile-binned HMM exp vs unexp ---")
    for q in d["test2"]["quartiles"]:
        e, u = q["expected"], q["unexpected"]
        print(f"   {q['quartile']}  pi_range={q['pi_range']}")
        print(f"     EXPECTED   n={e['n']:>5d}  peak_at_true={fmt(e['peak_at_true'])}  total={fmt(e['total'])}  "
              f"dec_acc={fmt(e['dec_acc'])}  mean_pi={fmt(e['mean_pi'])}  pred_err={fmt(e['mean_pred_err'])}  run_len={fmt(e['mean_run_length'])}")
        print(f"     UNEXPECTED n={u['n']:>5d}  peak_at_true={fmt(u['peak_at_true'])}  total={fmt(u['total'])}  "
              f"dec_acc={fmt(u['dec_acc'])}  mean_pi={fmt(u['mean_pi'])}  pred_err={fmt(u['mean_pred_err'])}  run_len={fmt(u['mean_run_length'])}")
        print(f"     Δpeak={q['delta_peak']:+.4f}  Δdec={q['delta_dec']:+.4f}")
    print()
    print("--- TEST 3: pi-matched-pair HMM (driver=expected) ---")
    t3 = d["test3"]
    print(f"   n_pairs={t3['n_pairs']}  matching: mean|Δpi|={t3['matching_quality']['mean_abs_pi_diff']:.4f}  "
          f"max|Δpi|={t3['matching_quality']['max_abs_pi_diff']:.4f}")
    print(f"   mean pi exp={t3['matching_quality']['mean_pi_exp']:.4f}  "
          f"mean pi unexp={t3['matching_quality']['mean_pi_unexp']:.4f}")
    print(f"   EXPECTED   peak={t3['expected']['peak_at_true']:.4f}  total={t3['expected']['total']:.4f}  dec={t3['expected']['dec_acc']:.4f}  run_len={t3['expected']['mean_run_length']:.3f}")
    print(f"   UNEXPECTED peak={t3['unexpected']['peak_at_true']:.4f}  total={t3['unexpected']['total']:.4f}  dec={t3['unexpected']['dec_acc']:.4f}  run_len={t3['unexpected']['mean_run_length']:.3f}")
    print(f"   Δpeak={t3['delta_peak']:+.4f}  Δtotal={t3['delta_total']:+.4f}  Δdec={t3['delta_dec']:+.4f}")
    print()
    print("--- BONUS: run-length stats (per-trial spread proxy) ---")
    bl = d["bonus_run_length"]["overall"]
    print(f"   expected_overall:  mean={bl['expected']['mean']:.3f}  median={bl['expected']['median']:.1f}  std={bl['expected']['std']:.3f}")
    print(f"   unexpected_overall: mean={bl['unexpected']['mean']:.3f}  median={bl['unexpected']['median']:.1f}  std={bl['unexpected']['std']:.3f}")


def main():
    print("\n" + "#" * 80)
    print("# Task #7: Full L2/3 activity characterization (R1+R2 seed 42)")
    print("#" * 80)
    print()
    for label, fname in JSONS.items():
        path = RESULTS / fname
        if not path.exists():
            print(f"!! MISSING: {path}")
            continue
        with open(path) as f:
            d = json.load(f)
        if label == "matched_3row_ring":
            report_3row_ring(d)
        elif label == "matched_hmm_ring_sequence":
            report_hmm_seq(d, label)
        elif label == "matched_hmm_ring_sequence_tightexp":
            report_hmm_seq(d, label)
        elif label == "matched_probe_3pass":
            report_3pass(d)
        elif label == "v2_confidence_dissection":
            report_dissection(d)


if __name__ == "__main__":
    main()
