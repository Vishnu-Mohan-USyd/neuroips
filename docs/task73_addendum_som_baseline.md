# Task #73 Addendum — SOM Baseline Root-Cause Evidence (debugger2)

Discovered during Task #74 Fix C investigation: L2/3 SOM r_som ≈ 505/unit (max 3176) on Phase-2 step_3000 ckpt with task weights zeroed. L23E is fine (r=0.216, max 3.4). Always-on Phase-2 issue, not a regression.

## Hypotheses & verdicts (numeric record)

H1 (fan-in miscalibration): **CONFIRMED**
- w_l23_som_raw_init = -1.0
- w_l23_som_raw_final = +8.0000
- w_l23_som_eff_init = softplus(-1) = 0.3133
- w_l23_som_eff_final = softplus(+8) = 8.0003
- w_l23_som_eff_rowsum_mean = 2048 (design ≈ 80, → 25.6× inflation)
- contribution_to_som_drive = 443 of 475 total (local=443, fb=32)

H2 (theta_som adaptation): **FALSIFIED**
- theta_som_mean = 1.0
- theta_som_init = 1.0
- delta = 0.0
- Architectural: `L23SOM` extends `InhibitoryPopulation` which stores a fixed float `target_rate_hz`, not an adaptive Parameter. Cannot drift.

H3 (softplus linear regime): **CONFIRMED**
- drive_mean = 475.40
- theta_mean = 1.0
- ratio = 475.40 (max 6778.75)
- At this ratio, softplus ≈ identity → r_som ≈ drive passes through unattenuated.

H4 (divisive stabilizer): **MISSING (CONFIRMED)**
- div_stabilizer_present = F
- Evidence: src/v2_model/layers.py:670-705 — forward only subtracts scalar target_rate_hz; no PV→SOM, no divisive term, no homeostatic cap.

## Synthesis

- Primary cause: H1 — W_l23_som_raw drifted from -1.0 to +8.0 during Phase-2 iSTDP.
- Amplifier: H3 — softplus saturates to linear at 475× threshold; drive passes through.
- Architectural enabler: H4 — no divisive stabilizer means W-drift is unbounded.

Compound magnitude: 25× (W inflation) × 18× (r_l23 above design 0.012 baseline) ≈ 450× excess drive vs original design operating point (matches observed 475×).

## Fix options (biologically grounded, coder's call)
1. PV→SOM divisive normalization (Pfeffer 2013) — architectural fix addressing H4.
2. iSTDP target-rate realignment (Vogels 2011) — plasticity fix preventing runaway W.
3. Turrigiano synaptic scaling on W_l23_som (Turrigiano 2008) — slow homeostatic cap. **Lead's preference: option 3.**

## Reproduction
- Script: `scripts/v2/_debug_task74_som_baseline.py`
- Log: `logs/task74/som_baseline_s42.log`
- Input ckpt: `checkpoints/v2/phase2/phase2_task70_s42/phase2_s42/step_3000.pt`
- Command: `python3 scripts/v2/_debug_task74_som_baseline.py`
