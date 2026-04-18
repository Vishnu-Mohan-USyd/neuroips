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

## Stage 0 gate — pending

(Rates 2-8 Hz, tuning FWHM 30-60 deg, H-baseline-quiet, g_total calibrated, 3 seeds sign-consistent 3/3.)

## Stage 1 gate — pending

(H bump persistence 200-500 ms; MI(leader, H_state_+500ms) > 0 for H_R;
 analogous rotation MI for H_T; no runaway; 3 seeds.)

## Stage 2 gate — pending

(Cue-H orientation selectivity d >= 0.2; cue-alone evokes H bump in >= 80 pct valid trials;
 H_R recurrent weights unchanged; 3 seeds.)
