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
modules. Commit `a99a8a7` (`test(sprint-3): component-level validators`).

| Validator | Assays | Result | Key numbers |
|---|---|---|---|
| `validate_nmda_channel.py`      | 3  | 3/3 PASS | tau_nmda=49.95 ms; V_1/2=-20.50 mV; NMDA:AMPA charge ratio=0.58 |
| `validate_per_channel_inh.py`   | 2  | 1/2 PASS | local suppression: 114->0 Hz; WTA margin=0.03 (FAIL, structural) |
| `validate_v1_ring.py`           | 3  | 3/3 PASS | smooth span=15 pA, cliff span=0 pA, monotone=yes |
| `validate_stimulus_schedules.py`| 14 | 14/14 PASS | Richter balanced 10/pair; Tang deviant frac=0.142 (expected 0.142) |
| `validate_plasticity.py`        | 5  | 5/5 PASS | LTP=+0.015, LTD=-0.294; NMDA deposit=w*amp=0.250 nS (exact) |

Totals: 26 PASS / 1 FAIL across 5 modules.

The single FAIL (WTA under balanced 2-channel drive) is a structural
gap: the current per-channel + broad-pool inh cannot enforce WTA when
two channels receive symmetric Gaussian drive. Stage-1 MI does not
depend on this property (the Richter cue is unambiguous, Tang blocks
are deterministic), so it is captured as a regression target rather
than a Stage-1 blocker. Remediation options (Mexican-hat cross-channel
inh, SFA on H_E, or cross-channel E->inh recurrence) are deferred to a
future sprint.

Run any validator:

    python -m expectation_snn.validation.validate_<name>

## Stage 2 gate — pending

(Cue-H orientation selectivity d >= 0.2; cue-alone evokes H bump in >= 80 pct valid trials;
 H_R recurrent weights unchanged; 3 seeds.)
