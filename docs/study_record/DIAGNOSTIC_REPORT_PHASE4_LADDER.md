# DIAGNOSTIC REPORT — Phase-4 ladder: s=0.04 cell verification + non-monotone trained-M
Debugger, 2026-08-19. Two-part commission. Checkpoint/log measurement only; zero training runs.

---
## PART A — s=0.04 joint-pass cells: **REAL** (both)

Probe: `probes/probe_ladder_verify_s0p04.py` → `probe_ladder_verify_s0p04_report.json`.

- **C1 config**: both endpoint checkpoints carry pred_inhib_strength 0.04 / σ 4.0 / rnn_tanh /
  seed 8 / step 8000, correct alpha per arm; both training.jsonl `run_start` lines match.
- **C2 provenance** (state sha in the harness's own convention, byte-exact mirror of
  train_sweep.py:327–335): each endpoint's sha == its OWN run's `alpha_complete` sha
  (dampening 04b771c1…, sharpening 9d929d2b…); each arm's `alpha_start` loaded/common sha ==
  its OWN run's `pretrain_complete` sha (4c5b1a32…). The two arms' pretrain shas equal EACH
  OTHER — correct: pretrain is α-independent and the environment is proven bitwise-deterministic;
  the endpoints differ as they must.
- **C3 independent recompute** (official measurement core `phase4_endpoint_eval.measure`
  verbatim; pipeline E0-anchored earlier at <1e-12 to the frozen anchor / official / A4):
  dampening M_auc_ratio **0.29606407963526854 exact (abs diff 0.0)**, center 0.14363 / flank
  0.49991 / H 0.19907 all 0.0; sharpening flank_ratio **0.8278569927266332 exact**, center
  1.18947 / H 0.99074 all 0.0. Sole nonzero diff anywhere: 5.4e-9 float32 rounding on a stored
  non-bar mean-rate field.

---
## PART B — the non-monotone trained-M column, explained

Probe: `probes/probe_part_b_nonmono.py` → `probe_part_b_nonmono_report.json`.
Comparators: original frozen α0.5 nets, seeds 8–11 (S2_confirm, frozen_gate_decision_rnn.json).

### The decomposition that isolates the anomaly

Per cell (all inference counterfactuals with the official measure(); Mo0 = 0.3320623 anchor):

| s | trained M | weights-only M (s→0) | adaptation at s=0 | direct term |
|---|---|---|---|---|
| 0.02 | 0.2889 | 0.3146 | +0.0174 | 0.0257 |
| 0.03 | 0.2500 | 0.2866 | +0.0455 | 0.0366 |
| 0.04 | 0.2961 | **0.3500** | **−0.0180** | 0.0540 |
| 0.05 | 0.2475 | 0.3020 | +0.0300 | 0.0545 |

**The direct-subtraction term is monotone in s (0.0257 → 0.0366 → 0.0540 → 0.0545) — the
mechanism's dose physics behaves exactly as proven.** The entire non-monotonicity lives in the
SETTLED WEIGHTS: the four runs converged to states holding different amounts of activity
(weights-only M 0.2866–0.3500), with s=0.04's state above even the original (0.3500 > 0.3321).
That is why it beat the zero-adaptation ceiling: the ceiling assumed adaptation ≥ 0; measured
adaptation at s=0.04 is −0.0180 (the retrained state holds MORE activity than the original).

### Hypothesis verdicts

| hypothesis | verdict | evidence |
|---|---|---|
| Hb1 distinct gain-configuration attractor at s=0.04 | **RULED OUT** | settled gains virtually identical across all four cells (k −3.41…−3.55, som_margin 1.81–1.84 — one family); profile shape r vs original ≥ 0.969 everywhere, s=0.04 the CLOSEST (0.9927); H all in-band. Same circuit configuration; the difference is fine W_fb/GRU structure, not a regime. |
| Hb2 deterministic path sensitivity within a non-settling regime | **CONFIRMED** (the explanation) | (i) divergence starts in PRETRAIN and is graded with Δs: pretrain d_Wfb L2 vs the s=0.05 pretrain = 1.09 / 0.76 / 0.41 / 0 for s=0.02/0.03/0.04/0.05 — the s change reroutes training from step 1; (ii) the α=0.5 regime NEVER SETTLES: in every cell's last 2000 steps k swings 0.9–1.0 and the training activity proxy spans 9–12% min-max; (iii) the resulting settled-M scatter (±10% relative) matches the regime's natural solution variability measured on the ORIGINAL study's own seeds: original α0.5 M = 0.3321/0.3071/0.3092/0.3133 (seeds 8–11; 8% relative range) — and seed 8, the band's anchor, is the TOP of that distribution; (iv) between neighboring rungs the functional stimulus difference (Δblanket ≈ 0.003 rate units) is far smaller than the outcome differences — disproportionate response = sensitive dependence. |
| Hb3 energy-objective directional adaptation (higher s pre-pays energy → keep more activity) | **RULED OUT as driver** | adaptation is sign-flipping non-monotone (+0.0455 at s=0.03 vs −0.0180 at s=0.04); no monotone trend exists to attribute. |

**Conclusion:** trained-M(s) = a monotone direct-subtraction dose response SUPERPOSED with
±0.02–0.05 micro-solution scatter intrinsic to the α=0.5 training regime. The column's
non-monotonicity is the scatter, not a dose law, and not specific to the surround (the original
no-surround seeds scatter on the same scale).

### Is s=0.04 legitimate or fragile?

**Legitimate as a state; the risk is elsewhere.**
- Not artifactual (Part A: exact recompute, clean provenance).
- Not a knife-edge: inference-s fan on the s=0.04 weights is smooth and linear
  (M = 0.3083 / 0.3013 / 0.2961 / 0.2906 / 0.2849 at s' = 0.030/0.035/0.040/0.045/0.050,
  slope ≈ −0.011 per 0.01s — it passes the M bar measured anywhere in s' ∈ [0.03, 0.05];
  H/center/flank essentially flat across the fan).
- In-family everywhere: closest profile to the original of all four cells, same gain
  configuration, topology preserved, H in-band, all bars passed with margin.

**The real risk, quantified:** the P3_M pass margin at s=0.04 is +0.0138 (0.2961 vs 0.2823),
which is SMALLER than the demonstrated between-run settled-M scatter (±0.02–0.05 across the
ladder; 0.025 range across the original's own seeds). Also relevant: the band is anchored on
original seed 8 (0.3321), the maximum of its own seed distribution — original seeds 9/10/11
would pass the band by only 0.025–0.031 themselves. So "M in band at s=0.04" is expected to be
a seed-dependent property, not because the cell is bogus but because settled activity in this
regime varies by more than the pass margin across equivalent runs.

**PREDICTION (labeled; multi-seed is the pre-registered arbiter):** seeds 9/10/11 at s=0.04
(dampening) will land trained-M scattered by ±0.02–0.03 around ≈ 0.28–0.30; substantial chance
(order one-in-three to one-in-two) that at least one seed falls below the 0.2823 bar. The
sharpening arm at s=0.04 (flank margin 0.828 vs 0.85 = 0.022, seed-invariance 0.003 at s=0.05)
is predicted much safer.

### Limits (stated, not glossed)

- The within-run stopping-time contribution (would step-7000 weights give a different M of the
  same order?) is INFERRED from the tail wander (k swings ~1.0, activity span ~10%) — it cannot
  be directly measured because no mid-run weight snapshots exist for these cells. Cross-RUN
  scatter, which is what the multi-seed decision needs, IS directly measured (4 ladder cells +
  4 original seeds).
- Why the s=0.03 path settled low (+0.0455) and the s=0.04 path high (−0.0180) has no further
  decomposable cause at checkpoint level — that is what "sensitive dependence within one solution
  family" means operationally.

Artifact shas recorded in the probe reports. All numbers trace to commands run this session.
