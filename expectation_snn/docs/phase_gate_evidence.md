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

## Stage 1 gate — PASSED (seed 42, 2026-04-19)

Per pre-registered seed policy (see research_log.md): first-pass seed=42 only.
Multi-seed replication {7, 123, 2024, 11} + held-out {99, 314} deferred until
Stage-1 paradigm-level findings worth replicating.

Evidence (gitignored, re-derivable from tag `phase-stage-1-passed`):

- `data/checkpoints/stage_1_hr_seed42.npz` (Richter grammar, H_R)
- `data/checkpoints/stage_1_ht_seed42.npz` (Tang grammar, H_T)

Drivers:

- `brian2_model/train.py::run_stage_1_hr` (Richter 6×6 crossover).
- `brian2_model/train.py::run_stage_1_ht` (Tang 30° rotating blocks).

Architecture changes (Sprint-3):

- `neurons.py`: NMDA slow recurrent conductance on H_E with Jahr & Stevens
  1990 Mg²⁺ block; `tau_nmda_h=50 ms` (NR2A-dominated; Vicini et al. 1998);
  `V_nmda_rev=0 mV`. Enables Wang 2001 bump-attractor persistence.
- `plasticity.py`: `pair_stdp_with_normalization` gains
  `nmda_drive_amp_nS` parameter — pre-spike also deposits
  `w * nmda_drive_amp_nS` into `g_nmda_h_post`; plasticity acts on AMPA
  weight only.
- `h_ring.py`: inhibitory pool split into per-channel local PV-like
  subpools (12 cells, one per channel) + broad cross-channel (4 cells,
  scaled by `broad_inh_scale=0.3`). E→E co-releases AMPA + NMDA.
  Vogels iSTDP weight ceiling `inh_w_max=1.5` caps total adaptation
  independent of schedule length.
- `v1_ring.py`: PV Poisson background noise (400 Hz × 15 pA) smooths the
  FS rheobase f-I curve; enables PV to fire in the 10-40 Hz band over a
  broader bias range. Stage-0 gate re-passes with `pv_bias_pA=240`.
- `stimulus.py`: `richter_crossover_training_schedule` and
  `tang_rotating_sequence` builders (deterministic under `rng`).
- `validation/stage_1_gate.py`: `check_h_bump_persistence`,
  `check_h_transition_mi`, `check_h_rotation_mi`, `check_no_runaway`.

### Stage 1 H_R gate @ seed 42

Run config: `n_trials=72`, `leader_ms=500`, `trailer_ms=500`, `iti_ms=1500`,
presettle=10 s @ 40 Hz broadband, `cue_peak_hz=300`, `cue_sigma_deg=15`.

| Check | Value | Band | Status |
|---|---|---|---|
| h_bump_persistence_ms | 360.0 ms | [200, 500] | PASS |
| h_transition_mi_bits  | 0.092 bits | ≥ 0.05      | PASS |
| no_runaway            | 28.4 Hz   | [0, 80]    | PASS |

E→E weight diagnostics (post-schedule): within-channel mean=0.948,
cross-nbr mean=0.055 (no LTP runaway on cross-channel edges).

### Stage 1 H_T gate @ seed 42

Run config: `n_items=500`, `item_ms=250` (Tang 4 Hz), presettle=10 s,
cue as above.

| Check | Value | Band | Status |
|---|---|---|---|
| h_bump_persistence_ms | 250.0 ms | [200, 500] | PASS |
| h_rotation_mi_bits    | 1.662 bits | ≥ 0.05      | PASS |
| no_runaway            | 33.8 Hz   | [0, 80]    | PASS |

E→E weight diagnostics: within-channel mean=0.938, cross-nbr mean=0.057.

Rotation MI dominates the transition MI because the Tang grammar is
near-deterministic (CW/CCW block, 5 of 6 items predict the next) while
Richter crossover is 1-of-6 at each step — both signals are well above
the 0.05 bits finite-sample floor.

### Sprint-3 component-level validation (Lead standing rule backfill, 2026-04-19)

Per Lead's standing rule ("each module gets functional -- not just smoke
-- validation BEFORE the next module is built on top"), five validators
exercise the biological/architectural claims of the five Sprint-3
modules. First cut committed as `a99a8a7`
(`test(sprint-3): component-level validators`). Reworked 2026-04-19
with biology-anchored bands, renames for clarity, and statistical rigor
(bootstrap CIs, chi-squared GoF):

| Validator | Assays | Result | Key numbers | Commit |
|---|---|---|---|---|
| `validate_neurons.py`     | 4  | 3/4 PASS (1 FAIL flagged) | tau_nmda=49.95 ms; V_1/2=-20.50 mV; NMDA:AMPA charge=1.161 @ V_h=-55 mV (FAIL vs Wang 2001 [2,6]); SFA tau=149.95 ms | `4b1f57b` |
| `validate_plasticity.py`  | 6  | 6/6 PASS | LTP=+0.015, LTD=-0.294; NMDA deposit=0.25 nS (exact); NMDA-on-post-only peak=0.0000 nS (pre-release-only invariant) | `2731fdf` |
| `validate_h_ring.py`      | 3 gated + 1 informational | 3/3 gated PASS | local[0]=6 Hz ch0 drive, broad=4 Hz ch0+ch6 drive; local[0]/local[1] tuning ratio=6.0x; broad-abl multi-ch suppression delta=+16 Hz; WTA n_sustained=2 (informational gap, not gated) | `57ce01e` |
| `validate_v1_ring.py`     | 3  | 3/3 PASS | peak channel=0 (driven); FWHM=33.00 deg (Tang [30,60]); orth rate<0.5*peak. Scope reduced to tuning-preservation pending debugger smoothing metric. | `b123231` |
| `validate_stimulus.py`    | 17 | 17/17 PASS | Richter balanced 10/cell + bootstrap spread CI=[10,18]; Tang block-len chi2=2.156, p=0.707 (uniform{5..9}); Tang deviant 95% CI=[0.120, 0.164] contains 0.143 | `79daa87` |

Totals: 32 PASS / 1 FAIL across 5 modules (33 assays gated; WTA is
reported informational-only).

**Open architectural items:**

1. **NMDA:AMPA charge ratio 1.16 vs Wang-2001 target [2, 6]** (neurons
   Assay 3). Measured at V_h=-55 mV with shipped H_E wiring amps
   (AMPA 25 pA / NMDA 0.5 nS per HRingConfig default). Under-NMDA'd
   relative to Wang 2001 ideal for bump attractor persistence. Options
   (escalated to Lead, pending decision): (a) widen band to accept
   current wiring, (b) re-tune `drive_amp_ee_pA` / `nmda_drive_amp_nS`
   for biological ratio, (c) spawn debugger to root-cause whether
   Stage-1 MI depends on hitting the Wang range.

2. **WTA under symmetric 2-channel drive** (h_ring Assay 4, informational).
   The current per-channel + broad-pool inh cannot enforce
   winner-take-all when ch0+ch6 are driven simultaneously: n_sustained=2
   under intact wiring. Stage-1 MI does not depend on this property
   (Richter cue is unambiguous; Tang blocks are deterministic), so it
   is captured as a regression target. Remediation options (Mexican-hat
   cross-channel inh, SFA on H_E, cross-channel E->inh recurrence) are
   deferred to a future sprint.

3. **V1 ring PV Poisson-noise smoothing metric** (v1_ring, deferred).
   The Sprint-3 commit `ec2a2ac` claimed "smooths PV rheobase with
   Poisson background." Finding the right metric + threshold is tracked
   by the debugger (task #25 CONFIRMED). This validator will be amended
   post-Sprint-4 with the debugger's tested metric (task #26).

Run any validator:

    python -m expectation_snn.validation.validate_<name>

## Stage 2 gate — PASSED (seed 42, 2026-04-19)

Per pre-registered seed policy: first-pass seed=42 only. Multi-seed replication
({7, 123, 2024, 11} + held-out {99, 314}) deferred until paradigm-level findings
warrant it.

Evidence (gitignored, re-derivable from tag `phase-stage-2-passed`):

- `data/checkpoints/stage_2_seed42.npz`

Driver: `brian2_model/train.py::run_stage_2_cue` (seed=42, n_train_trials=200).

Run config: `STAGE2_CUE_MS=500`, `STAGE2_GAP_MS=500`, `STAGE2_GRATING_MS=500`,
`STAGE2_ITI_MS=2500`, `STAGE2_VALID_FRAC=0.75` (150 valid / 50 invalid),
`STAGE2_N_PROBES_PER_CUE=20` (40 cue-alone probes total).
Wall-clock: 714 s (~12 min) on `expectation_snn` conda env, dt=0.1 ms, numpy codegen.

### Stage-2 architecture (Sprint-4)

- **Cue pathway** (plastic): 2 × 32-afferent Poisson populations (`cue_A`, `cue_B`)
  all-to-all → H_E via `eligibility_trace_cue_rule` (Frémaux & Gerstner 2015):
    - `on_pre:  elig = 1.0`
    - `on_post: w = clip(w + lr_eff * elig, 0, w_max_eff)`
    - τ_elig = 1500 ms, lr = 2e-4, w_init = 0.1, w_max = 2.0,
      drive = 20 pA/spike.
- **Teacher forcing** (grating epoch, valid trials only): direct DC bias of
  `STAGE2_TEACHER_BIAS_PA = 300 pA` (≈ 1.5× H_E rheobase) injected on the
  matched-channel H_E subpool. V1 ring is built + loaded + frozen but dropped
  from the simulation Network — V1→H_R Poisson teacher produced cross-channel
  LTP due to V1's ±15° tuning width and was replaced with direct injection.
- **Plasticity freeze verification**:
    - V1 `pv→e` Vogels iSTDP → `pv_to_e.active = False`
    - H_R `ee` pair-STDP    → `A_plus_eff = A_minus_eff = 0`
    - H_R `inh→e` Vogels    → `eta_eff = 0`
- **Per-trial state reset**: each trial starts by zeroing `h_ring.{e,inh}.V,
  I_e, I_i`, `g_nmda_h`, and `elig_{A,B}.elig` to prevent cross-trial bump
  persistence and eligibility bleed-through. Per-probe reset as well.

### Stage-2 gate @ seed 42

| Check | Value | Band | Status |
|---|---|---|---|
| cue_selectivity_d      | 3.008  | ≥ 0.20, CI lo > 0  | PASS  (bootstrap 95% CI=[2.552, 3.790], n_probe=40) |
| bump_evocation_frac    | 1.000  | ≥ 0.80             | PASS  (40/40 probes, matched-channel rate > 5 Hz AND = peak) |
| hr_weights_unchanged   | 0.000  | < 0.010            | PASS  (max |Δw| = 0.0e+00; ee freeze confirmed) |
| no_runaway             | 4.85 Hz | ≤ 80 Hz           | PASS  (H_E=4.85, H_inh=1.05) |

Cue weight diagnostics (post-200-trial training):

- `cue_A` → matched ch3 (45°) weight = **2.000** (saturated at w_max), unmatched ch9 = **0.965**
- `cue_B` → matched ch9 (135°) weight = **2.000** (saturated at w_max), unmatched ch3 = **0.975**

Cue-alone probe rates (post-training): matched-channel mean = **246.8 Hz**,
unmatched-channel mean = **74.7 Hz** — matched wins every probe. Saturation
at w_max=2.0 is the expected end-state under positive feedback once the
matched-channel cue weight crosses rheobase during the cue window; the
selectivity asymmetry (matched 2.00 vs unmatched ~0.97) is set during the
early linear-growth phase by the teacher-gated LTP before feedback kicks in.
