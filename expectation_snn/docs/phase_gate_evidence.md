# Phase-gate evidence log

## Phase 0

Commit timestamp + evidence for each gate as phases complete.

### Phase 0 deliverables

- [x] Orphan branch `expectation-snn-v1h` created (root-commit 20ee8c1)
- [x] `env.yml` single Brian2 env (Py 3.12, numpy >=2.0, brian2 2.10.1, brian2tools 0.3)
- [x] Repo skeleton per plan sec 10
- [x] Conda env `expectation_snn` created successfully (brian2 2.10.1, brian2tools 0.3,
      numpy, scipy, matplotlib, h5py, scikit-learn, pytest). brian2cuda deferred — optional,
      commented out in env.yml pending CUDA toolchain verification.
- [x] Brian2 smoke test output committed (`scripts/smoke_test.log`).
- [x] Tang 2023 paradigm numbers verified from PMC full text (PMC9981605) and logged in
      `docs/research_log.md`. Plan assumptions (30° / 4 Hz / 250 ms / no ISI / 5–9-block)
      all confirmed. Added rotation direction (CW or CCW) and 50% contrast / 0.034 cpd
      details to Methods log for stimulus builder reference.

## Stage 0 gate — PASSED (seed 42, 2026-04-19)

Per pre-registered seed policy (see research_log.md): first-pass seed=42 only. Multi-seed
replication {7, 123, 2024, 11} + held-out {99, 314} deferred until Stage-0 calibration is
re-run at full scale or on a meaningful parameter change.

Evidence: `data/checkpoints/stage_0_seed42.npz` (gitignored, re-derivable from tag
`phase-stage-0-passed`).

Run config: bias_probe_ms=500, istdp_settle_ms=10000, final_probe_ms=1000, wall-clock 20.2 s.
Driver: `brian2_model/train.py::run_stage_0`.

Gate results:

| Check | Value | Band | Status |
|---|---|---|---|
| v1_e_rate_band        | 4.42 Hz | [2.00, 8.00]   | PASS |
| v1_pv_rate_band       | 39.00 Hz | [10.00, 40.00]  | PASS |
| v1_som_rate_band      | 4.00 Hz | [2.00, 6.00]   | PASS |
| v1_tuning_fwhm_deg    | 36.38° | [30.00, 60.00] | PASS |
| h_baseline_quiet      | 0.00 Hz | [0.00, 1.00]   | PASS |
| h_pulse_response      | 224 spikes ch0 vs 0 ch6 | pulsed/orth > 2 | PASS |
| no_runaway            | 39.00 Hz | [0.00, 80.00]  | PASS |

Calibrated values (seed 42):

- V1_E tonic bias: 32.8 pA (bisection converged @ target 5 Hz → measured 4.42 Hz).
- PV→E synaptic weights (Vogels iSTDP, 10 s settling): mean 0.057, max 0.323.
- PV tonic bias: 302 pA (hand-tuned; FS cliff at ~300-305 pA).
- SOM tonic bias: 130 pA.
- Stimulus Gaussian σ: 22° (gave FWHM 36.4° vs target 30-60°).

## Stage 1 gate — pending

(H bump persistence 200-500 ms; MI(leader, H_state_+500ms) > 0 for H_R;
 analogous rotation MI for H_T; no runaway; 3 seeds.)

## Stage 2 gate — pending

(Cue-H orientation selectivity d >= 0.2; cue-alone evokes H bump in >= 80 pct valid trials;
 H_R recurrent weights unchanged; 3 seeds.)
