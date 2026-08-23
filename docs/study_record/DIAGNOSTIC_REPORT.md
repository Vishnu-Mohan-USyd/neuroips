# DIAGNOSTIC REPORT — flank-sharpening collapse root cause
Debugger, 2026-08-19. Commission: prove WHY the rung-1 surround run (pred_inhib_strength s=0.5,
pred_inhib_sigma_channels σ=4.0, seed 8, α=0) collapses (dead ring 0.82/0.83, H 0.47/0.67,
center boost gone, flanks 0.13/0.08), single-variable, before any fix. No fix is implemented here.

---
## Failure and reproducer

Failure: kill-class gate/endpoint signature on
`/home/vishnu/scratch/flank_sharpening_20260819/runs/predinhib_s0p5_sig4/seed_8`
(coder's rung-1 run): gate step 4000 H 0.4676, dead_ring 0.8175, center_ratio 0.9827,
flank_ratio 0.1333; endpoint step 8000 H 0.6667, dead_ring 0.8297, center_ratio 0.9248,
flank_ratio 0.0815, effective k 0.5457→0.1658.

Reproducer: the coder's artifacts themselves (deterministic assay). My measurement pipeline
first reproduced the coder's endpoint numbers exactly before any other cell was trusted
(E0 gate): my eval of the same checkpoint = H 0.6666666865, dead 0.8297324777,
center 0.9248366928, flank 0.0815496870 — matches `endpoint_report.json`.

## Method and provenance

All probes under `/home/vishnu/scratch/flank_sharpening_20260819/probes/` (pre-authorized),
cuda:0, sequential single-process, MemAvailable ≥25 GB asserted in-script, 8 GB RSS watchdog on
both training probes, `PYTHONHASHSEED=0 python3 -B`. Frozen roots read-only throughout; the
frozen lib was never modified (independent reimplementation + one instrument harness copy).

| artifact | sha256 (first 16) |
|---|---|
| probes/reimpl.py (independent forward reimplementation) | 9d735f9287e91bd0 |
| probes/probe_unit_h1.py + probe_unit_h1_report.json | d16ae87e17c2d8d8 / 21ea6f07a2f09cf7 |
| probes/probe_ladder.py + probe_ladder_report.json | a206f575c9e985f4 / 803e5398549f0d4f |
| probes/probe_trained_endpoints.py + report | 3936a228aa422905 / 663dbbe9d6c4c463 |
| probes/harness_s0p05/train_sweep.py (instrument copy, ONE line differs: s 0.5→0.05) | 9db8f975531b55a8 |
| coder harness reused verbatim for the arm-only run | fbb4172be88b6606 (= RUN_LOG post-edit sha) |

Training probes (2, pre-registered with the lead before running, predictions fixed in advance):
- E4b arm-only: frozen healthy pretrain (S2_confirm seed_8, file sha 5542b43f…, copy verified
  bitwise) + coder's harness → in-log `pretrain_resume` at 3000/3000 (zero pretrain steps),
  `pretrain_complete` state sha 926c53fb…b574 = the frozen state bitwise → arm 8000 with surround on.
- E2c dose: s=0.05, σ=4.0, fresh pretrain, seed 8, single variable vs rung 1 = strength only.

The reimplementation used for all budget decompositions was validated BITWISE against the frozen
lib before use (probe_unit_h1 T4/T5: lib `l23` == reimpl rate on every test; full unroll
`forward_seq_tuned` preds AND rates max-abs-diff 0.0 on both the collapsed endpoint net and the
frozen healthy α0 net over the full 216-trial assay batch).

---
## Hypotheses tested

| # | Hypothesis | Verdict | One-line evidence |
|---|---|---|---|
| H1 | Lib surround math wrong (sign/normalization/application point/wraparound/baseline silence) | **RULED OUT** | Kernel == independent recomputation (max diff 5.6e-9, rows sum 1, symmetric, wraparound exact, values match DESIGN table); delta-fb pre-relu difference == 0.5·F·K[:,j] to float32 precision at F=1/5/35, j=0/10/35; fb=0 and t0 outputs bitwise-identical to a strength-0 net; reimpl bitwise == lib on real nets. |
| H2 | Magnitude overdose: pooled subtraction at the TRUE f scale exceeds off-center excitation | **CONFIRMED** (root cause) | Measured blanket s·(K@f) ≈ 0.35–0.44 on every channel vs off-center drive ≤0.46; floors 72–83% of channels instantly on ANY weights with zero training; single-variable dose s 0.5→0.05 flips collapse→clean (H 0.67→1.00). |
| H3 | Gain escape: k-fall causes the collapse | **RULED OUT** | Temporal precedence: full collapse already present at arm step 0 with k bitwise at init 0.545718789100647 (H 0.125, dead 0.770); the healthy NO-surround arm's k falls the same way and deeper (0.5457→0.0366 vs surround 0.1658); H improves 0.468→0.667 while k keeps falling. |
| H4 | Pretrain interaction: surround during pretrain wrecks the representation | **RULED OUT as necessary/primary** (real but secondary contributor) | Arm-only run on the bitwise healthy pretrain collapses the same (H 0.537, dead 0.829 vs coder 0.667/0.830); surround-pretrained weights with surround OFF are only mildly degraded (H 0.343 vs healthy 0.431, ring fully alive); at s=0.05 pretrain tracks the frozen healthy pretrain to the 3rd–4th decimal (task 0.1639 vs 0.1646 at step 3000). |

---
## Proven causal chain

**1. At the network's real operating point, feedback evidence f is ~30 units of mass, not ~1.**
Measured at the continuation-A final step (216-trial mean): Σf per trial = 24.9–31.9 across all
five weight sets probed; per-channel f reaches 27.2 (healthy endpoint center) and 34.2 max.
DESIGN §"center tax" arithmetic (s·K(0)·f ≈ 0.050 = "7.2% of g3") implicitly used f≈1.
DESIGN's own risk note ("f can reach 36p−1 ≫ 1 … right absolute value is an empirical question")
is the branch that materialized.

**2. Row-normalized pooling turns that mass into a near-uniform subtraction blanket.**
K (σ=4) is symmetric with unit row sums ⇒ unit column sums ⇒ mean_i (K@f)_i = Σf/36 exactly.
Measured mean s·K@f per channel: 0.368 (healthy-pretrain+surround), 0.346 (arm0), 0.382
(endpoint), 0.438 (healthy-endpoint+surround) — each equals s·Σf/36 to 3 decimals. The
"43% of inhibitory mass in the flank band" design framing is true of the KERNEL ROW but
irrelevant at the operating point: with Σf ≈ 27 spread through K, EVERY channel receives
≈0.35–0.44, regardless of its offset.

**3. The blanket exceeds total excitation everywhere outside the center ±3, so relu floors the
ring — mechanically, before any training.** Off-center excitation is drive only (f≈0 there):
drive = 0.46 at ±4, 0.24 at ±5, 0.11 at ±6, ≤0.004 beyond ±9. Aligned per-offset budget on the
coder's arm-step-0 checkpoint (k still at init): pre_relu at ±5 = −0.256, ±6 = −0.353,
±9 = −0.155; fraction of channels with pre_relu<0 = 0.770 = the dead ring. Bolting s=0.5 onto
the UNTRAINED-with-surround healthy pretrain weights instantly gives dead 0.721, H 0.065; onto
the healthy α=0 ENDPOINT: dead 0.762, H 0.069. Three different weight sets, zero training steps,
same floor ⇒ weight-independent mechanical overdose.

**4. The floored/flattened profile breaks the readout/prediction stack ⇒ H collapse.** The
population-vector readout and RNN were trained on graded ring profiles (healthy: 1.50 → 1.36 →
1.06 → 0.52 → 0.27 → 0.09 across offsets 0–6). Under the blanket the profile becomes a 4–6
channel plateau (0.74/0.73/0.73/0.62 at offsets 0–3) + zeros. H drops 0.431→0.065 on healthy
pretrain weights with zero training. The f profile also flattens (f(0): 6.95→2.15), reducing
center feedback excitation — a secondary self-reinforcing loop.

**5. The 8000 arm steps ADAPT to the blanket rather than dying from it.** Task loss improves
monotonically (0.214→0.161), H recovers 0.125 (arm0) → 0.468 (4000) → 0.667 (8000). The
trained weights come to DEPEND on the surround: endpoint with s switched off at inference drops
to H 0.319 (and the arm-only endpoint to 0.282). The dead ring never bakes into weights — s→0
at inference instantly revives every floored channel (dead 0.830→0.0000 at the endpoint,
0.829→0.0000 arm-only, 0.770→0.0000 at arm0) — it is re-imposed mechanically every forward pass.

**6. Causal proof by single-variable dose (E2c).** Changing ONLY s (0.5→0.05; same σ=4.0, seed,
harness family, steps): pretrain tracks the frozen healthy run (task 0.1639 vs 0.1646, next_ce
1.1599 vs 1.1649 at step 3000 — the +26%/+0.30-nat pretrain degradation at s=0.5 also vanishes,
proving that degradation was the same overdose, dose-dependent); endpoint H = **1.0000**,
center_ratio 1.192, flank_ratio 0.789, M 0.158, k trajectory ≈ healthy (0.544→0.0499 vs healthy
0.5457→0.0366). Predicted blanket 0.05·Σf/36 ≈ 0.044 (measured 0.044) sits below the flank-band
drive (0.05–0.12 tax against 0.11–0.46 drive) — the mechanism becomes the intended graded tax
instead of a floor. Outcome flips from total collapse to task-perfect flank-suppressed
sharpening on the strength knob alone.

## Falsified alternative readings (corrections to the evidence trail)

- **G1's "effective k FALLING 0.546→0.207" was misread as neutralization.** The healthy
  no-surround arm does the same, harder: k 0.5457→0.2545 (step 500)→0.0679 (4000)→0.0366 (8000)
  (frozen sweep s8_t1p0_e0p0 log). The surround run keeps k HIGHER than healthy at every step.
  k-fall is the normal arm dynamic under this objective, not a surround pathology. DESIGN §
  "neutralization signature (k collapsing + f flattening)" therefore cannot be used as a kill
  criterion against k alone.
- **The coder's "pretrain references bitwise in-family" check verified nothing about the
  mechanism.** `reference_values` (harness line 233) calls `net.l23(l4, zeros)`; with fb=0 the
  subtraction is identically zero (proven bitwise in T3). A_ref/R_ref/sigma_train equality is a
  data/init identity check only — necessary, not sufficient. Process flag for future gates.
- **dead_ring's exactly-zero definition over-counts under ANY subtractive term.** Far-ring
  channels (|offset| ≥ 7–9) have baseline activity 1e-6..1e-26 (Gaussian ff tails); any nonzero
  blanket floors them to literal 0.0 without functional consequence. At s=0.05 the
  subtraction/drive crossover sits at |offset| = 8 (0.0220 vs 0.0148), flooring 21/36 = 0.583 of
  channels — the metric reads 0.583 while H = 1.0 and the ±0–7 profile is graded and healthy.
  The gate bar "dead_ring < ~0.5" implicitly assumed no subtractive mechanism; with one present
  it needs an activity-floor-aware definition (e.g. fraction of channels with baseline activity
  above some ε that are zeroed) — validator/lead call, flagged here only.

## Key numbers (all cells: 216-trial assay, gate_eval conventions, my pipeline == coder's at E0)

| cell (weights × inference-s) | H | dead | center | flank | k |
|---|---|---|---|---|---|
| frozen healthy pretrain, s=0 (as trained) | 0.431 | 0.000 | 1.793 | 0.929 | 0.5457 |
| frozen healthy pretrain, s=0.5 bolted on | 0.065 | 0.721 | 0.943 | 1.236 | 0.5457 |
| coder pretrain (s=0.5-trained), s=0.5 | 0.125 | 0.770 | 1.246 | 0.742 | 0.5457 |
| coder pretrain, s=0 switched off | 0.343 | 0.000 | 1.649 | 1.272 | 0.5457 |
| coder arm steps 1500→4000 (s=0.5) | 0.292→0.468 | 0.805→0.818 | 1.03→0.98 | 0.26→0.13 | 0.263→0.207 |
| coder endpoint 8000 (s=0.5) | 0.667 | 0.830 | 0.925 | 0.082 | 0.1658 |
| coder endpoint, s=0 switched off | 0.319 | 0.000 | 1.281 | 1.183 | 0.1658 |
| frozen healthy α0 endpoint, s=0 | 0.995 | 0.000 | 1.178 | 0.979 | 0.0366 |
| frozen healthy α0 endpoint, s=0.5 bolted on | 0.069 | 0.762 | 0.566 | 0.351 | 0.0366 |
| E4b arm-only endpoint (healthy pretrain, s=0.5 arm) | 0.537 | 0.829 | 0.977 | 0.089 | 0.1873 |
| E4b endpoint, s=0 switched off | 0.282 | 0.000 | 1.361 | 1.103 | 0.1873 |
| E2c dose endpoint (s=0.05 throughout) | **1.000** | 0.583* | **1.192** | **0.789** | 0.0499 |
| E2c dose endpoint, s=0 switched off | 0.944 | 0.000 | 1.233 | 0.972 | 0.0499 |

*metric artifact — see falsified-readings section; crossover at |offset|=8, functional band alive.

Full per-offset budget tables (drive, g3·f, g4·som, s·K@f, pre_relu, rate, f per offset) for six
cells: `probes/probe_ladder_report.json` and `probes/probe_trained_endpoints_report.json`.

## What any fix must mechanistically address (no implementation — coder's job)

1. **Calibrate the subtraction to the measured f scale, not f≈1.** The governing quantity is the
   blanket s·Σf/36 (exact identity for a symmetric row-normalized kernel) against the off-center
   drive floor (0.11–0.46 in the flank band ±4–6). s=0.5 exceeds that floor 2–4×; s=0.05 sits
   under it and already produces genuine flank suppression (0.789×) with center enhancement
   (1.192×) and H 1.0 in the single-seed dose probe. Any chosen s must keep s·Σf/36 below the
   flank-band drive at the operating point (Σf ≈ 25–32 measured).
2. **Do not chase larger suppression by raising s** — that reintroduces the floor (proven at
   0.5). If more flank suppression than pure-s scaling gives is wanted, the mechanism must change
   WHERE the subtraction lands (shape of the pooled term relative to baseline), not its global
   magnitude.
3. **The dead_ring gate needs an activity-floor-aware definition** before it can arbitrate any
   subtractive-mechanism run (otherwise every such run "fails" on 1e-26-baseline channels).
4. Observed-not-required: the s=0.05 probe endpoint incidentally meets protocol acceptance
   criteria 1–3 (flank 0.789 ≤ 0.85, center 1.192 ≥ 1.15, H 1.000 ≥ 0.95) on a single seed.
   This is a DIAGNOSTIC observation, not a validated fix: it ran in my probe pipeline, criterion
   4 (mechanism-development audit) and the validation suite were not run, and the ladder ruling
   (s ladder 0.5→0.25/1.0) pre-dates this evidence. Lead's call.

## Remaining unknowns

- The exact collapse threshold in s between 0.05 (clean) and 0.5 (collapsed) is unmeasured; the
  budget arithmetic predicts trouble once s·Σf/36 crosses the ±5–6 drive (≈0.11–0.24, i.e.
  s ≈ 0.15–0.3), but no run tests this. Not needed for the root-cause verdict.
- Whether s=0.05's flank suppression (0.789×) is stable across seeds — multi-seed is
  post-acceptance territory per protocol, not probed.

All claims above trace to a command and output in this session; raw JSON artifacts and scripts
are in `/home/vishnu/scratch/flank_sharpening_20260819/probes/` with shas listed in Method.
