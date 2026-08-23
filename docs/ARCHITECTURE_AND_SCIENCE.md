# Architecture and Science — the surround-inhibition study

This is the one document. It explains (1) the network architecture end-to-end,
(2) the surround mechanism precisely, (3) its biological grounding, (4) the
training recipe and determinism guarantees, (5) the science story — honestly,
including the failure that taught the dose arithmetic and the family claim
that did not confirm, (6) the measurement conventions behind every number, and
(7) the provenance table binding every claim to an artifact and sha256.

Every number in this document was re-verified against the study record
(checkpoints, evaluation reports, the validator's independent re-derivations)
at packaging time. Verbatim copies of the governing study documents are in
[`study_record/`](study_record/INDEX.md).

---

## 1. The network, end-to-end

The model is a small recurrent predictive circuit over a 36-channel
orientation ring (180° of orientation, 5° per channel), implemented in
`src/tuned_emergence_lib.py` (`SimpleTunedNet`) and driven by
`src/train_sweep.py`. Data flow per timestep:

**Stimulus → L4 (fixed encoder).** An orientation θ (degrees) becomes a
36-channel circular-Gaussian population code, `l4_code(θ)` in
`src/simple_net.py`: channel preferred orientations at 5° spacing, tuning
width σ = 12°, circular distance on the 180° ring. Handcoded and never
trained.

**L4 → L2/3 (`SimpleTunedNet.l23`).** The layer-2/3 rate vector is computed
from the current L4 code and the *previous step's* feedback signal `fb`:

```
drive      = ff_gain · (l4 @ K_ff.T)                    ff_gain = 1.6, K_ff: circular Gaussian, σ = 1.1 ch
fb_pos     = relu(fb)
g0..g4     = softplus(circ_raw)                          5 learned scalar gains, init softplus(0) = ln 2 ≈ 0.693
vip        = relu(g0 · fb_pos)                           VIP-like: feedback-driven disinhibitor
som        = relu(g1 · fb_pos − g2 · vip)                SOM-like: feedback-driven, VIP-inhibited
pred_inhib = s · (fb_pos @ K_σ.T)                        THE SURROUND TERM (§2); s = pred_inhib_strength
rate       = relu(drive + g3 · fb_pos − g4 · som − pred_inhib)
rate       = rate / (1 + c · (rate @ K_lc.T))            divisive local competition, K_lc: σ = 2.0 ch
```

Two further terms exist in the code but are configured OFF in this study:
`pred_feature_supp` (strength 0.0) and the firing-rate adaptation state
(`adapt_strength` 0.0 — the unroll calls the update, but it is a no-op).
Rate saturation is off (`rate_saturation_r_max` 0.0). All `K_*` kernels are
row-normalized circular Gaussians over the 36 channels
(`local_circular_matrix(σ)`); these are sign-constrained rate variables, not
identified interneuron dynamics.

The scalar-gain VIP/SOM motif collapses, for analysis, into one **effective
feedback gain** `k = g3 − g4·max(g1 − g2·g0, 0)`; at init k = 0.5457. The
divisive-competition gain `c = softplus(raw)` is trainable with init ln 2 and
σ fixed at 2.0 ch; it trains during pretraining and is **frozen during the α
arms** (`freeze_local_comp`, byte-verified unchanged pretrain→final in every
study cell).

**L2/3 → RNN → feedback.** `h_t = RNNCell_tanh(rate_t, h_{t−1})` (36→64; the
study regime passes `--recurrent-cell rnn_tanh`; the config default field says
`gru` and is overridden on the CLI in every study run). The prediction is
`pred_t = W_fb(h_t)` (Linear 64→36, raw logits). The feedback *signal* the
next step receives is the **posterior–prior excess**:

```
f = relu(36 · softmax(pred) − 1)
```

— the excess of the implied posterior over the uniform prior. It is
nonnegative, exactly zero for an uninformative prediction, and grows to
36·p − 1 ≈ 35 on a confident channel. Its actual trained scale, **Σf ≈ 25–32
per step at the study's operating points, concentrated on few channels**, is
the single most important magnitude in this study (§5.2).

**Unroll order (matters for measurement, §6):** at each step t the circuit
computes `rate_t` from `l4_t` and `f_{t−1}`, then updates the RNN and emits
`f_t` for step t+1. **Step t=0 is therefore feedback-silent** (`f_{−1} = 0`):
every feedback-attached mechanism — including the surround — is structurally
OFF at the baseline timestep the assay ratios divide by.

**Readout.** Population-vector decoder: rates are projected onto the ring's
(cos, sin) resultant, normalized, and re-expanded into 36 logits scaled by a
learned gain (init 8.0, softplus-parameterized).

---

## 2. The surround mechanism, precisely

### 2.1 The entire change: two config constants

The study's complete diff against the frozen trainer
(`heatmap_sweep_20260818/harness/train_sweep.py`, sha `cdd71a11…`, §7) is two
lines of `MODEL_CONFIG` in `src/train_sweep.py`:

```
"pred_inhib_strength":        0.0  → 0.05        (dose s; 0.04 in the family ladder)
"pred_inhib_sigma_channels":  0.65 → 4.0         (footprint σ, channels)
```

No library, loss, CLI, or assay edit. The hook itself (`pred_inhib` in
`l23()`) always existed in the frozen library — shipped dormant at s = 0 with
a uselessly narrow default σ. At s = 0 the term is *exactly* zero, and σ is
end-to-end inert: the validator proved forward outputs, all parameter
gradients, and input gradients bitwise-identical between σ = 0.65 and σ = 4.0
at s = 0 (VERDICT.md, Check 3). `pred_inhib_strength` alone carries the
entire effect.

### 2.2 The math

`K_σ = local_circular_matrix(4.0)` is a row-normalized circular Gaussian over
the 36-ring (rows sum to 1; σ in channel units, 1 ch = 5°). Per channel i,
with the feedback mass on predicted channel(s) j:

```
rate_i = relu( drive_i + k·f_i − s·(K f)_i − … )
```

For a center-concentrated f this is a **difference-of-Gaussians on the
feedback path**: the predicted channel keeps its direct excitation `k·f_j`
and loses only the small self-pooled term `s·K(0)·f_j`, while every
non-predicted channel receives pure subtraction `s·K_ij·f_j` with no
offsetting excitation. Boost-narrow / inhibit-broad — the classic sharpening
asymmetry, here gated by feedback.

Kernel footprint at σ = 4.0 (N = 36, row-normalized; from DESIGN.md §3,
recomputed):

| | K(0) | K(3) | K(4) | K(5) | K(6) | 2·ΣK(3..6) = mass in the ±15–30° flank band |
|---|---|---|---|---|---|---|
| σ = 4.0 (chosen) | 0.0997 | 0.0753 | 0.0605 | 0.0457 | 0.0324 | **0.428** |
| σ = 0.65 (shipped default) | 0.6135 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 |

At σ = 4.0, **42.8% of every unit of feedback-recruited inhibition lands
exactly in the assay's ±15–30° flank band**; the shipped σ = 0.65 default is
narrower than the excitation itself — a center-attacking configuration (the
known failure mode), which the σ change converts into a flank-targeting one.

**The blanket identity (the dose arithmetic).** Because K is row-normalized,
the ring-mean of the subtraction is exactly

```
mean_i s·(K f)_i = s · Σf / 36
```

independent of where f is concentrated. This identity — verified exact to
measurement precision on the study's cells (0.034678 and 0.034404 at
s = 0.05) — is what makes s a *dose*: at the trained scale Σf ≈ 25–32, the
mean subtraction is ≈ 0.7·s in rate units, to be compared against off-center
drive ≤ 0.46. It predicted both the s = 0.5 catastrophe and the workable
s ≈ 0.05 range (§5).

**Gating.** The term is multiplied into `relu(fb)`: it fires only when the
network is confidently predicting — exactly the assay's measured
"adapted/continuation" state — and is structurally silent at the t0 baseline
(§1 unroll order). The mechanism can therefore move flank/own-baseline
ratios, and an s→0 inference counterfactual (A4, §6) cleanly isolates its
contribution.

**What is fixed vs learned.** σ is bio-fixed anatomy (§3), never trained —
also removing any suspicion that a learned width was fit to the assay bands;
s is a fixed constant with no trainable path in the library (plain float).
All *learned* adaptation happens around the fixed mechanism: `circ_raw` (the
k-balance), `W_fb` (what gets predicted, hence where inhibition lands), and
the recurrent cell set the effective operating point.

---

## 3. Biological grounding

The mechanism transplants a well-established cortical circuit — **top-down
feedback recruiting broadly-pooled, SOM-mediated subtractive surround
suppression with a spared/facilitated center** — from visual space into
orientation space. Six primary sources, all read in full during the study
(local copies in the study dir's `papers/`; the two load-bearing statistics
sets were re-verified verbatim against the full texts by the independent
validator — VERDICT.md Check 4):

1. **Adesnik H, Bruns W, Taniguchi H, Huang ZJ, Scanziani M (2012).** *A
   neural circuit for spatial summation in visual cortex.* Nature 490:226–231.
   doi:10.1038/nature11526. SOM interneurons lack surround suppression
   (SI 0.09±0.06) and sum over a ~4× broader footprint than pyramidal cells
   (preferred size 86±3° vs 22±2°), fed by horizontal L2/3 axons (241±85% of
   PC-level excitation vs only 17±5% from L4); silencing SOMs reduces
   pyramidal surround suppression by 30±10% (p=0.00022) and facilitates
   larger-than-preferred responses by 74±19% **while leaving preferred-size
   responses unchanged (−7±7%, p>0.45)** — the center-sparing,
   surround-targeting incidence this design copies structurally.
2. **Zhang S, Xu M, Kamigaki T, Do JPH, Chang W-C, Jenvay S, Miyamichi K,
   Luo L, Dan Y (2014).** *Long-range and local circuits for top-down
   modulation of visual cortex processing.* Science 345:660–665.
   doi:10.1126/science.1254126. Top-down feedback axons produce center
   facilitation (+0.17±0.02 at 0 μm, p=4e−16) plus surround suppression
   (−0.15±0.03 at 200 μm, p=4e−6) by recruiting local interneurons; SOM+
   cells carry the surround arm, VIP+ the center arm — the same VIP/SOM motif
   the model's scalar gains already lump into k, completed here with the
   missing broad SOM arm. **Note 37: 200 μm ≈ 20° of visual angle — the
   quantitative anchor for σ = 4 channels = 20°.** Their discussion
   explicitly proposes the same circuit operating "in stimulus feature
   space", licensing the orientation-domain transplant.
3. **Nurminen L, Merlin S, Bijanzadeh M, Federer F, Angelucci A (2018).**
   *Top-down feedback controls spatial summation and response amplitude in
   primate visual cortex.* Nature Communications 9:2281.
   doi:10.1038/s41467-018-04500-5. Inactivating V2→V1 feedback *reduces*
   in-RF responses (−32.0±6.03%) and *increases* near-surround responses
   (+29.2±7.14%; SI 0.21→0.006) — intact feedback simultaneously boosts the
   center and suppresses the near surround; their recurrent model reproduces
   the effect set with one mechanism, asymmetric horizontally-pooled
   (SOM-like) inhibition driven by the same feedback that excites E cells —
   the computational abstraction implemented here as `g3·f − s·K f`.
4. **Ma W-p, Liu B-h, Li Y-t, Huang ZJ, Zhang LI, Tao HW (2010).** *Visual
   representations by cortical somatostatin inhibitory neurons — selective
   but with weak and delayed responses.* J Neurosci 30:14371–14379.
   doi:10.1523/JNEUROSCI.3248-10.2010. SOM cells are orientation-tuned as
   strongly as excitatory neurons (unlike untuned PV) — justifying an
   orientation-*tuned* Gaussian surround kernel rather than uniform
   inhibition; their delayed, facilitating, distal-dendrite-targeting
   responses mark them as gates of later-arriving intracortical/feedback
   input — supporting attachment of the pooled inhibition to `fb_pos`, not to
   the feedforward `drive`.
5. **Ben-Yishai R, Bar-Or RL, Sompolinsky H (1995).** *Theory of orientation
   tuning in visual cortex.* PNAS 92:3844–3848. doi:10.1073/pnas.92.9.3844.
   The canonical ring model J(Δθ) = −J0 + J2·cos(2Δθ) — net Mexican-hat
   connectivity in orientation space generating sharp, contrast-invariant
   tuning — the 50-year lineage for lateral-inhibition-in-orientation-space
   sharpening, instantiated here on the feedback path.
6. **Carandini M, Heeger DJ (2012).** *Normalization as a canonical neural
   computation.* Nat Rev Neurosci 13:51–62. doi:10.1038/nrn3136. The
   suppressive pool "typically responds to a broader set of stimuli than the
   summation field" — the broader-pool-than-drive requirement that is
   definitional for surround suppression; cited for the computation, not a
   specific synapse (their caveat that V1 divisive phenomena are not
   uniformly GABA_A-dependent is noted; the causal circuit evidence is
   refs 1–3).

**Honest scope note (stated in the study record, repeated here):** the cited
circuit evidence is a SPATIAL surround; this model instantiates it in
ORIENTATION space. The transplant is explicit and is grounded in Zhang
2014's feature-space proposal and the Ben-Yishai ring-model lineage — it is a
principled abstraction, not a claim that these papers measured orientation-
domain surround suppression.

---

## 4. Training recipe and determinism

**Task.** Random orientation trajectories on the ring (velocity/acceleration
random walk, clamped), presented as L4 codes; the network is trained to
predict the *next* channel and to keep a clean population vector. From
`task_energy_losses`:

```
task   = 0.5 · CE(next-step logits, next channel) / ln 36  +  0.5 · population_vector_loss / 2
energy = mean(rates) / R_ref
```

`R_ref` is the mean feedback-free L2/3 response of the just-pretrained
network to the 36 canonical single-orientation stimuli (a per-run activity
normalizer, computed once at arm start).

**Schedule.** Per (seed, α) cell: fresh task-only pretrain **3000 steps**,
then one arm of **8000 steps** minimizing `(1−α)·task + α·energy`.
Two regimes, identical architecture, differing ONLY in α:
- **Sharpening:** α = 0.0 (pure task).
- **Dampening:** α = 0.5 (energy-pressured).

Within a seed, the two arms' fresh pretrains are state-bitwise IDENTICAL
(α only touches the arm) — validator-verified on all s = 0.04 cells. Official
seed 8; confirmation seeds 9/10/11. One cell ≈ 3.5 min wall on an RTX 5090
(11,000 steps total), peak RSS ≈ 2 GB.

**Determinism.** Every study run: `PYTHONHASHSEED=0 python3 -B`, `cuda:0`,
harness-set `torch.use_deterministic_algorithms(True)`,
`cudnn.benchmark=False`, full seeding. Under these conditions a (seed,
config, device) triple reproduces **bitwise**: the study's A/A control
re-trained the frozen no-surround α=0.0 seed-8 cell through the study harness
with s zeroed and reproduced the frozen final AND pretrain states
sha-for-sha over all 11,000 steps (VERDICT.md Check 3) — proving the study
harness contains no behavioral delta beyond the two constants.

**Mechanism trained-in, not bolted-on.** The surround config is carried from
pretrain step 0 through the arm and into measurement (config-carried in
`run_start`, the pretrain checkpoint, and the final checkpoint of every
cell). The kernel is a `persistent=False` buffer — absent from `state_dict`,
rebuilt from checkpoint config at load — so measurement uses exactly the
trained mechanism, and the A4 counterfactual (§6) is a pure config flip on
identical weights.

---

## 5. The science story — honestly

### 5.1 The question

The pre-existing result: under α = 0.0 the trained circuit "sharpens" its
population profile after predictive feedback — but entirely by **boosting the
center** (+18–23%); flanks sat near baseline (flank ratio ≈ 0.97). Real
cortical sharpening also *suppresses* flanks. Goal (pre-registered,
PROTOCOL.md): make flank suppression emerge — flank_ratio ≤ 0.85 with
center_ratio ≥ 1.15, next-step accuracy H ≥ 0.95, an alive ring (vitality),
and an s→0 counterfactual showing the mechanism does the work — while
changing as little as possible. The chosen design (DESIGN.md RANK 1) is the
two-constant surround of §2.

### 5.2 First contact: the s = 0.5 overdose collapse

The design's initial dose s = 0.5 (reasoned as "moderate fraction of the
center gain", implicitly assuming f ≈ O(1)) collapsed the circuit: endpoint
H 0.6667, dead ring 83% of channels, center 0.9248, flank 0.0815. The
debugger's root cause (DIAGNOSTIC_REPORT.md, single-variable, confirmed):
**magnitude overdose.** The feedback scale is not O(1) — trained Σf reaches
25–34 — so by the blanket identity (§2.2) the mean subtraction was 0.35–0.44
per channel against off-center drive ≤ 0.46: relu floors 72–83% of the ring
*mechanically, on any weights, with zero training* (shown on three weight
sets). Competing explanations were killed: gain-escape ruled out by temporal
precedence (collapse already full at arm step 0 with k bitwise at init);
pretrain-interaction ruled out as primary (arm-only on healthy pretrain
collapses identically). The dead ring never bakes into weights — s→0 at
inference instantly revives every floored channel (dead 0.830 → 0.000).
A single-variable dose flip s 0.5 → 0.05 (blanket ≈ 0.044, below flank-band
drive) turned total collapse into a clean result: H 0.67 → 1.00.

Lesson encoded in this repo's comments: the dose knob must be sized against
the **trained feedback scale** via `s·Σf/36`, not against the gain constants.

### 5.3 The validated headline: s = 0.05 sharpening (verdict GO)

At s = 0.05, σ = 4.0, α = 0.0 — all four pre-registered bars pass on the
official seed and every confirmation seed, independently re-derived at full
float precision by the validator with **zero mismatches** (VERDICT.md
Check 1):

| seed | H (≥0.95) | center (≥1.15) | flank (≤0.85) | vitality | A4 flank (s→0) | A4 H |
|---|---|---|---|---|---|---|
| 8 (official) | 1.0 | 1.1923 | 0.7886 | pass | 0.9716 | 0.9444 |
| 9 | 0.9815 | 1.2035 | 0.7873 | pass | 0.9696 | 0.8981 |
| 10 | 0.9954 | 1.2160 | 0.7864 | pass | 0.9680 | 0.8750 |
| 11 | 0.9954 | 1.2156 | 0.7858 | pass | 0.9684 | 0.8889 |

Flank spread across seeds 0.786–0.789 — strongly seed-invariant. The stretch
goal (flank ≤ 0.75) was not met on any seed, as reported. Three evidence
lines make the claim mechanism-attributable and architecture-clean:

- **A4 counterfactual:** zeroing s at inference on the trained weights
  returns the flank to 0.968–0.972 (the no-surround healthy reference is
  ≈ 0.97) while the center boost *stays* (1.23–1.26): the surround, and
  nothing else, does the flank work.
- **Co-adaptation:** removing the surround at inference *degrades* the task
  (H 1.0 → 0.944 on seed 8; 0.875–0.898 on seeds 9–11) — the trained weights
  rely on it; it is load-bearing, not cosmetic.
- **A/A bitwise control** (§4): the harness carries no hidden deltas; s alone
  carries the effect; σ is inert at s = 0 (forward + gradients bitwise).

Task cost at endpoint: none (seed-8 H = 1.0 vs frozen healthy reference
0.9954). One design prediction was wrong and is recorded: flank suppression
was predicted to *deepen* as feedback sharpens; observed flat/slightly-up
(0.771 → 0.789 across the arm) — full-magnitude from the first snapshot,
because fixed connectivity is present from step 0 (validator G-R1 review).

### 5.4 The family question, and the s = 0.05 dampening failure

The user flagged a real confound: comparing sharpening-WITH-surround against
the original dampening-WITHOUT-surround compares two architectures. The
family claim worth having is: **one fixed architecture (s, σ), and α alone
switches the phenotype** — sharpening at α = 0.0, dampening at α = 0.5. So
the dampening regime was re-run under the identical s = 0.05/σ = 4.0
architecture, against pre-registered bands pinned from the frozen original
no-surround α = 0.5 seed-8 cell (reference values, reproduced from frozen
artifacts to ≤ 2.8e−17 before any retrain was read: center 0.149572, flank
0.559042, H 0.194444, M 0.332062, mean continuation rate 0.055348; bars:
P1 center ≤ 0.35, P2 center < flank, P3 H and M within ±15% relative —
M ∈ [0.282253, 0.381872], H ∈ [0.165278, 0.223611], P4 alive).

Result at s = 0.05: the dampening *phenotype* is intact — P1 pass (center
0.0997), P2 pass, P3_H pass (H 0.2083), P4 pass — but **P3_M FAILS**:
M = 0.2475 vs floor 0.2823 (25.5% below the original's 0.3321). The run was
stopped and handed to the debugger per protocol; no remedy was attempted.

**Debugger's decomposition (closed 2×2, weights × inference-s, exact
reproduction of all anchors — DIAGNOSTIC_REPORT_PHASE4_M.md):** the M
shortfall is 64–74% **direct measurement-time subtraction** (the blanket
identity, exact: s·Σf/36 = 0.0347/0.0344) and only 26–36% trained-weight
adaptation, with a NEGATIVE interaction (the retrained weights partially
*compensate* for the surround). **No regime change:** profile correlation
r 0.970 vs the original, topology preserved, H slightly better. The
architecture does not break dampening — the fixed dose taxes the activity
number the M band measures.

### 5.5 The joint dose ladder and the s = 0.04 window

Since the M tax is ≈ linear in s while sharpening needs s large enough to
clear flank ≤ 0.85, a joint window was sought: s ∈ {0.02, 0.03, 0.04}, both
regimes, seed 8, all bars verbatim, all six cells run regardless of early
results:

| s | sharpening flank (≤0.85) | sharpening verdict | dampening M (floor 0.28225) | dampening verdict | joint |
|---|---|---|---|---|---|
| 0.02 | 0.9047 | FAIL (too weak) | 0.28894 | PASS | — |
| 0.03 | 0.8665 | FAIL | 0.24996 | FAIL | — |
| 0.04 | 0.8279 | **PASS** (H 0.9907, center 1.1895) | 0.29606 | **PASS** (all P1–P4) | **JOINT PASS** |
| 0.05 (§5.3/§5.4) | 0.7886 | PASS | 0.24748 | FAIL (P3_M) | — |

Two honest findings inside this table:

- **The trained M is non-monotone in s** (0.289 → 0.250 → 0.296 → 0.247),
  and s = 0.04 *contradicted* the debugger's pre-registered central
  prediction (robust FAIL; even the zero-adaptation ceiling 0.2805 was below
  the bar; measured 0.2961 exceeded that ceiling). The deviation was flagged,
  and the debugger's follow-up (DIAGNOSTIC_REPORT_PHASE4_LADDER.md) first
  verified both s = 0.04 cells are REAL (exact recomputation from
  checkpoints), then explained the non-monotonicity: the **direct
  subtraction is monotone in s** (0.0257 → 0.0366 → 0.0540 → 0.0545 across
  0.02/0.03/0.04/0.05) but sits superposed on **±0.02–0.05 settled-weights
  scatter**: the α = 0.5 objective never fully settles (k swings 0.9–1.0,
  activity ±10% over the last 2000 steps), and the training path is
  dose-sensitive from pretrain onward, so each cell lands on a different
  member of a scatter family. s = 0.04's weights-only state landed high
  (0.3500, above even the original), s = 0.03's low.
- That scatter is not an artifact of the surround: the **original no-surround
  α = 0.5 family itself spans M 0.3071–0.3321 across seeds 8–11** — with the
  band's seed-8 anchor at the TOP of its own family's range.

### 5.6 Multi-seed at s = 0.04: family claim NOT CONFIRMED (Outcome O2)

Seeds 9/10/11 × both regimes at s = 0.04, bars verbatim, validated
independently with zero exact-precision mismatches (VERDICT.md Addendum 2):

**Sharpening (α = 0.0): 4/4 PASS**, again strongly seed-invariant
(flank 0.8279 / 0.8253 / 0.8245 / 0.8240, spread 0.0038; H ≥ 0.9676; center
≥ 1.1895; vitality all pass).

**Dampening (α = 0.5):** every bar passes on every seed EXCEPT the M band:

| seed | M | P3_M (floor 0.2822529581893272) | other bars |
|---|---|---|---|
| 8 | 0.29606 | PASS (margin +0.0138) | all pass |
| 9 | 0.26372 | **FAIL** (−6.6% rel) | all pass |
| 10 | 0.28201 | **FAIL** (short 2.47e−4 = 0.09% of the floor) | all pass |
| 11 | 0.30908 | PASS | all pass |

Joint seed verdicts 8 YES / 9 NO / 10 NO / 11 YES → **the family claim at
s = 0.04 is NOT CONFIRMED** under the pre-registered bands (Outcome O2), with
no band loosened and no seed dropped. That is the study's verdict and this
repository states it as such.

**Calibration caveat (recorded context — not a modification of the
verdict):** the M band is anchored to a *single* reference cell (the frozen
seed-8 original) that sits at the top of its own family's M range
(0.3071–0.3321); the original family's own floor margins (0.025–0.031) are
comparable to the demonstrated between-run scatter (±0.02–0.05), and seed-8's
s = 0.04 pass margin (0.0138) is smaller than that scatter. Under a
distribution-referenced criterion the picture could differ in either
direction — but per the outcome rules, that would require **fresh seeds under
a newly pre-registered criterion**, not a re-reading of these.

### 5.7 A4 evidence, both regimes

The s→0 inference counterfactual (recorded as evidence beside every cell,
never as a bar) shows the mechanism doing the work in both regimes at
s = 0.04: sharpening flank returns 0.8279 → 0.9730 (seed 8; 0.969–0.970 on
seeds 9–11) while dampening M returns 0.2961 → 0.3500 / 0.2637 → 0.3207 /
0.2820 → 0.3305 / 0.3091 → 0.3446 (seeds 8–11) — removing the surround at
inference pushes every dampening cell's M back to or into the original band.
A bonus structural finding: within each seed the two regimes' t0 baselines
are **bitwise identical** (max abs diff 0.0) — t0 is feedback-silent and the
arms share a bitwise-identical pretrain — so each family figure's two curves
differ only through the trained arm.

### 5.8 What is claimed, and what is not

**Claimed (validated GO):** enabling the dormant surround with two constants
(s = 0.05, σ = 4.0) produces real, seed-invariant, mechanism-attributable
flank suppression in the sharpening regime at zero endpoint task cost, with
the architecture proven clean by a bitwise A/A control and the attribution
proven by the s→0 counterfactual.

**Claimed (measured, honest):** s = 0.04 is the only tested dose where both
regimes pass all their bars on the official seed; sharpening at s = 0.04 is
robust across all four seeds.

**Not claimed:** family-level parity at a single fixed dose. The dampening
activity band at s = 0.04 is contested on 2 of 4 seeds (one by 6.6%, one by
0.09%) → Outcome O2, NOT CONFIRMED. The dampening *phenotype* (suppression
below baseline, center-below-flank profile shape, H band, vitality) is intact
on 4/4 seeds; the contested quantity is the retained-activity magnitude M
against a band anchored to a top-of-its-family single reference, in a regime
with demonstrated between-run scatter comparable to the margins involved.

---

## 6. Measurement conventions (what every number means)

All numbers come from the frozen assay battery (fixed 216-history stimulus
set; the validator reproduced the registered generator `torch.equal`-exact
and re-derived every headline number independently).

- **Ring geometry:** 36 channels × 5°; offsets are channel distances from the
  history's final (expected) channel, in degrees. Profile curves are plotted
  over offsets −60°…+60° (25 points).
- **Baseline vs adapted:** the *baseline* is the literal t = 0 response
  (feedback-silent by unroll order, §1) pooled over histories; the *adapted*
  curve is the continuation-A final-step response aligned to each history's
  final channel. Ratios are per-band adapted/baseline on the mean aligned
  profile — each cell is compared against its *own* baseline.
- **center_ratio:** offsets {−5°, 0°, +5°} (center bin ±1).
- **flank_ratio:** offsets ±{15°, 20°, 25°, 30°} (bins ±3..±6). The kernel's
  42.8% flank mass (§2.2) targets exactly this band — by *design* of the
  mechanism, while the criterion itself predates it (frozen assay).
- **H (next-step accuracy):** fraction of the 216 histories whose step-index-3
  prediction argmax equals the true final channel; granularity 1/216 ≈ 0.0046.
- **M (dampening retained activity):** frozen definition
  `whole_36_bin_expected_A_AUC_over_timestep0_AUC` — continuation-final
  aligned mean rate divided by t0 mean rate (the 36 bins × 5° AUC factors
  cancel). M < 1 = dampening; the P3 band asks it to stay within ±15% of the
  original architecture's value.
- **Vitality (A3 sense):** every channel within |offset| ≤ 10° of the mean
  aligned profile > 0.01, plus mean rate > 0.01 in the dampening P4 bar. (An
  all-36-channel floor would fail the healthy *original* — its far ring
  reaches 2.3e−10 — so the band-alive operationalization was pre-registered
  with that justification before any retrain was read.)
- **A4 counterfactual:** rebuild the network from the checkpoint's own config
  with `pred_inhib_strength = 0`, load the same `state_dict`, re-run the
  identical assay. Inference-only; evidence, never a bar.
- **Bars, verbatim:** sharpening — flank ≤ 0.85, center ≥ 1.15, H ≥ 0.95,
  vitality (stretch flank ≤ 0.75, not met, reported). Dampening — P1 center
  ≤ 0.35; P2 center < flank; P3 H ∈ [0.165278, 0.223611] and
  M ∈ [0.282253, 0.381872] (±15% rel of the frozen reference, band edges
  carried at full float precision); P4 mean rate > 0.01 and A3-band alive.
- **Precision discipline:** criterion values are recorded and compared at
  full float precision (the validator's re-derivations matched the study's
  reports with zero mismatches; the one recorded near-miss, seed-10
  M = 0.2820059371106222 vs floor 0.2822529581893272, is a real sub-band
  value, short by 2.470210787050009e−4, not a rounding artifact).

---

## 7. Provenance

Everything below lives on the study machine (reuben-ML); frozen roots are
read-only. sha256 throughout. Study docs are copied verbatim into
[`study_record/`](study_record/INDEX.md) with the same shas.

### 7.1 Frozen inputs (pre-study, read-only)

| artifact | path | sha256 |
|---|---|---|
| Base library (verbatim in `src/`) | `/home/vishnu/neuroips_rnn_recreation_20260808/repo/simple_net.py` | `511581a640526a9bdbfca9effc72f60420211ee3825d7449162667e81e716f74` |
| Tuned library incl. the `pred_inhib` hook (verbatim in `src/`) | `/home/vishnu/neuroips_rnn_recreation_20260808/repo/tools/tuned_emergence_lib.py` | `3024bf0718ba69231e60f6a807cde2bfda0e10218519f6c5b7319ae222110e7a` |
| Frozen trainer the harness descends from | `/home/vishnu/neuroips_analysis/heatmap_sweep_20260818/harness/train_sweep.py` | `cdd71a11cbd254aa452f3b60f4f9da4350fe9fd85f7dcdf95cd35513435c250e` |
| Frozen original α=0.5 seed-8 checkpoint (dampening reference) | `/home/vishnu/neuroips_runs/rnn_recreation_20260808/S2_plot/seed_8/alpha_0p5_final.pt` | `156cc0f2372c6abcd42dd0798ac012d94bf2f761f7e8a860fb5bcc8fbc70bc18` |
| Frozen M definition + reference metrics | `/home/vishnu/neuroips_runs/rnn_recreation_20260808/S2_confirm/frozen_gate_decision_rnn.json` | (frozen root; M definition + per-seed originals quoted in §5/§6) |

### 7.2 Harness states (the entire study = one constant's trajectory)

`src/train_sweep.py` here is the study harness at its official state. Diff
audits at every state change were recorded (each exactly 2 lines − / 2 lines
+, the `pred_inhib_*` constants only) in the study RUN_LOG.

| state | `pred_inhib_strength` / σ | sha256 |
|---|---|---|
| frozen original | 0.0 / 0.65 | `cdd71a11cbd254aa452f3b60f4f9da4350fe9fd85f7dcdf95cd35513435c250e` |
| study official (= `src/` lineage; current on disk) | 0.05 / 4.0 | `9db8f975531b55a86c54791c68908708403cd4df72a97591ce8199b1ec25937e` |
| ladder s=0.02 (transient) | 0.02 / 4.0 | `cff25df2…` (full sha in RUN_LOG diff-audit) |
| ladder s=0.03 (transient) | 0.03 / 4.0 | `7a65c9a0…` (full sha in RUN_LOG diff-audit) |
| ladder s=0.04 (transient, used twice) | 0.04 / 4.0 | `7eb46f6c2a3b22885574b3961ce97ba9a1224259dc6654075cc8421b0e25d821` |

### 7.3 Run dirs and endpoint checkpoints (root: `/home/vishnu/scratch/flank_sharpening_20260819/`)

| cell | run dir (under `runs/`) | final checkpoint sha256 |
|---|---|---|
| s=0.5 collapse rung (α0.0 s8) | `predinhib_s0p5_sig4/seed_8/` | (superseded rung; numbers in DIAGNOSTIC_REPORT.md) |
| s=0.05 α0.0 seed 8 (official) | `predinhib_s0p05_sig4/seed_8/` | `c0a72f6a528a1ce79a5f435ad6e51c14522d52dc635f047460ae7587af660b6e` |
| s=0.05 α0.0 seed 9 | `predinhib_s0p05_sig4/seed_9/` | `ea8d4269acf6ce27c6fc92aa23cc5e1a56fe9865093d8f31e52d05495ac4e1e9` |
| s=0.05 α0.0 seed 10 | `predinhib_s0p05_sig4/seed_10/` | `4cd285ace335effe8c5a3f6811c2806cfb0445ba03eb4230e2d1c274ece18cd4` |
| s=0.05 α0.0 seed 11 | `predinhib_s0p05_sig4/seed_11/` | `22061df1d95318edef78c843e2aad63620659f3021b7981909cece7b4b47fcc2` |
| s=0.05 α0.5 seed 8 (Phase 4) | `predinhib_s0p05_sig4_alpha0p5/seed_8/` | `a456fbccefd24c7bd59a5cb5e0e8d78ed3d7e644b0f0560bcf64d6cf49e6a1fe` |
| s=0.02 α0.0 / α0.5 seed 8 | `ladder_s0p02/alpha0p{0,5}_seed8/` | (ladder rungs; eval reports below) |
| s=0.03 α0.0 / α0.5 seed 8 | `ladder_s0p03/alpha0p{0,5}_seed8/` | (ladder rungs; eval reports below) |
| s=0.04 α0.0 seed 8 | `ladder_s0p04/alpha0p0_seed8/seed_8/` | `a4e112df37778928a98d93442421e6953be4cdc5bc56624fa166c0c1e67bf26a` |
| s=0.04 α0.5 seed 8 | `ladder_s0p04/alpha0p5_seed8/seed_8/` | `3a78945efd3d3110f7db625e3e8c61a5eb9eaa7ab7c95153df181bd83ae572cf` |
| s=0.04 α0.0 seed 9 | `ladder_s0p04/alpha0p0_seed9/seed_9/` | `a1c8d7b152f37452588d8b5089dfd40c260f3bcfb7e919d90d72f6a348a70409` |
| s=0.04 α0.5 seed 9 | `ladder_s0p04/alpha0p5_seed9/seed_9/` | `8109fa534361fe10fd33c7a15f2c8a20bb7da9f6e7d89462d56da0a13abcce10` |
| s=0.04 α0.0 seed 10 | `ladder_s0p04/alpha0p0_seed10/seed_10/` | `1161a00560791480d08e363b3abf7dcc015ec25f98967d83a52d52e9f308d1f3` |
| s=0.04 α0.5 seed 10 | `ladder_s0p04/alpha0p5_seed10/seed_10/` | `cdb9cf114a8c351c8a5480daa1404a58ade370ac509ced41c5ed92587074c0a0` |
| s=0.04 α0.0 seed 11 | `ladder_s0p04/alpha0p0_seed11/seed_11/` | `59fabd7d8f305a60ab6cabf7a030eb080d748b00f74e545f0796023408239194` |
| s=0.04 α0.5 seed 11 | `ladder_s0p04/alpha0p5_seed11/seed_11/` | `a43f58906ec53b54cafaa00db7ee08a1896821b777c7f29f0b98304dc48b6fac` |

Every checkpoint sha above was recomputed from disk at packaging time.

### 7.4 Evaluation reports (same root)

| report | sha256 |
|---|---|
| `phase4_reference_alpha0p5_seed8.json` (pinned dampening reference) | `2ddb0e8be4837a486e77886754139038104eaf1b42096e99479d09302fc97835` |
| `runs/predinhib_s0p05_sig4_alpha0p5/endpoint_report_seed8.json` | `7b8073b0670b565fead034b2a8f9578c13869b049f0a1f39b881193000978a50` |
| `runs/ladder_s0p02/eval_alpha0p0_seed8.json` / `…0p5…` | `eae6ab2a128e9e7f…` / `1c91b0e0ad36f7dd…` |
| `runs/ladder_s0p03/eval_alpha0p0_seed8.json` / `…0p5…` | `a2df9508a110b542…` / `04792fb26c3a7efc…` |
| `runs/ladder_s0p04/eval_alpha0p0_seed8.json` / `…0p5…` | `fdf48fea678a0529…` / `d01f88f9692866e3…` |
| `runs/ladder_s0p04/eval_alpha0p0_seed9.json` / `…0p5…` | `ce9b1724d39945d4…` / `cf97195f42194a94…` |
| `runs/ladder_s0p04/eval_alpha0p0_seed10.json` / `…0p5…` | `af8cd13545ccbc85…` / `b1f188018f21ffb4…` |
| `runs/ladder_s0p04/eval_alpha0p0_seed11.json` / `…0p5…` | `e986f5e6e8e87be0…` / `adee6b55b646b08e…` |

(16-hex prefixes shown for the twelve ladder reports for table width; each
report also embeds the sha of the checkpoint it measured, and the full values
are in the study RUN_LOG.)

### 7.5 Study documents (verbatim copies in `study_record/`, full shas there)

`PROTOCOL.md` `cf5035af…` · `DESIGN.md` `85286bba…` ·
`DIAGNOSTIC_REPORT.md` `da191414…` · `DIAGNOSTIC_REPORT_PHASE4_M.md`
`549c4b71…` · `DIAGNOSTIC_REPORT_PHASE4_LADDER.md` `6bed6dea…` ·
`VERDICT.md` `f5640239…` — see [`study_record/INDEX.md`](study_record/INDEX.md)
for full shas and roles. Validator's independent artifacts:
`/home/vishnu/neuroips_outputs/validator_flank_20260819T013110Z/`.

### 7.6 Delivered figures

| deliverable | dir | key sha |
|---|---|---|
| s=0.05 sharpening figures | `/home/vishnu/neuroips_outputs/flank_sharpening_20260819/` | (generator copy: `src/make_flank_sharpening_figs.py`) |
| s=0.04 family figures (both regimes, identical architecture) | `/home/vishnu/neuroips_outputs/family_s0p04_figs_20260822/` | `provenance.json` `9cffc787e94418efe9f690079347b9b8e6c2134b98dfdaa5af1148b48dc069bb` |

The family figures carry the study's honest status line verbatim: *"family
parity: 4/4 sharpening pass; dampening phenotype intact 4/4, activity band
contested on 2/4 seeds — verdict O2, see VERDICT.md sha f5640239…"*.
