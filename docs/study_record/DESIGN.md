# Phase 1 DESIGN — flank-suppressed sharpening via biologically plausible architecture
Researcher deliverable, 2026-08-19. Governing protocol: `PROTOCOL.md` in this directory.
Status of all code below: mapped READ-ONLY; no file outside this study directory was modified.

---

## 0. Executive summary

**Top pick: enable the existing feedback-recruited subtractive surround (`pred_inhib`) with a
broadened, bio-fixed orientation footprint** — a two-constant `MODEL_CONFIG` change in a byte-copy
of the proven trainer, zero library edits:

```
"pred_inhib_strength":        0.0  -> 0.5        # bio-fixed SOM-like surround gain
"pred_inhib_sigma_channels":  0.65 -> 4.0        # 4 ch x 5°/ch = 20° footprint (Zhang 2014, note 37)
```

This converts the L2/3 rate equation into a difference-of-Gaussians in orientation space on the
feedback path: **narrow center excitation** (`g3·fb`, the existing scalar-k motif) **minus broad
pooled inhibition** (`s·K_σ4·fb`), which is precisely the circuit that Adesnik 2012 / Zhang 2014 /
Nurminen 2018 established for SOM-mediated, feedback-recruited surround suppression with a spared
center. The mechanism is architectural (connectivity footprint), references no profile band in any
objective, is present in pretrain + arm training and in the assay (config flows through checkpoint
`tuned_net_config` automatically), and is maximally leveraged by the assay convention itself:
feedback is ~silent at the t0 baseline and strong at the continuation-A final step, so the surround
subtraction fires exactly in the measured state and not in its reference.

---

## 1. The regime being modified (verified from the frozen α=0.0 checkpoint)

Read from `/home/vishnu/neuroips_runs/rnn_recreation_20260808/S2_plot/seed_8/alpha_0p0_final.pt`
(`torch.load`, read-only):

| fact | value |
|---|---|
| seed / alpha / steps | 8 / 0.0 / 8000 arm (+3000 task-only pretrain) |
| recurrent_cell | `rnn_tanh` |
| feedback_mode | `posterior_prior_excess` (`center_feedback: False`) |
| task_weight | `None` → historical `1 − α` = 1.0 |
| freeze_local_comp | `True` (local_comp ln2 ≈ 0.6931, σ=2.0 ch, divisive, trainable-but-frozen) |
| pred_inhib | **OFF** (`strength 0.0`, σ 0.65) |
| readout | `population_vector`, `population_normalize: True` |

Established phenomenology (protocol §Established facts): α=0 gives center +18–23%, flanks only
−2 to −3%; any energy pressure ε≥0.2 attacks the center first.

---

## 2. Exact code-path map (file : line, all verified this session)

### 2.1 The model — `/home/vishnu/neuroips_rnn_recreation_20260808/repo/tools/tuned_emergence_lib.py` (READ-ONLY, 332 lines)

- **Kernel builder** — `local_circular_matrix(sigma_channels)`, lines 38–42: full 36×36
  nonnegative circular Gaussian over channel distance, **rows normalized to unit sum**
  (`w / w.sum(dim=1, keepdim=True)`). Row normalization ⇒ total inhibitory mass per source channel
  is constant in σ; broadening σ *redistributes* mass outward, it does not add mass.
- **Constructor** — `SimpleTunedNet.__init__`, lines 55–128. Mechanism arguments already exist:
  `pred_inhib_strength` (line 63, default 0.0), `pred_inhib_sigma_channels` (line 64, default
  0.65); `local_comp_*` (lines 71–75). `local_comp_strength_raw` is an `nn.Parameter` when
  `local_comp_trainable=True` (lines 107–109; softplus-inverse init).
- **Mechanism kernels are `persistent=False` buffers** — `pred_inhib_weight` lines 111–115,
  `adapt_weight` 116–120, `local_comp_weight` 121–125. Consequence: they are **not** in
  `state_dict`; they are rebuilt from config at construction. No stale-kernel trap when loading
  old state into a new-config net. (`ff_weight`, line 110, *is* persistent, but its σ is unchanged
  in every candidate.)
- **The L2/3 stage** — `l23()`, lines 149–173. The heart of the design:
  ```python
  drive   = self.feedforward(l4)                       # narrow fixed FF basis (σ_ff = 1.1 ch)
  fb_pos  = F.relu(fb)                                 # f = relu(36·softmax(W_fb h) − 1)
  g       = F.softplus(self.circ_raw)
  vip     = F.relu(g[0] * fb_pos)
  som     = F.relu(g[1] * fb_pos - g[2] * vip)
  pred_inhib = self.pred_inhib_strength * (fb_pos @ self.pred_inhib_weight.t())   # line 165
  ...
  rate    = F.relu(drive + g[3]*fb_pos - g[4]*som - pred_inhib - pred_feature_supp - adapt)  # line 168
  rate    = self.apply_local_competition(rate)         # line 169
  ```
  Note the existing SOM/VIP motif (`vip`, `som`) shares the *same per-channel footprint* as
  `fb_pos` — it modulates the scalar k, it cannot reach neighbors. `pred_inhib` is the **only**
  hook that spatially pools the feedback across channels before subtracting.
- **Divisive competition** — `apply_local_competition()`, lines 175–187:
  `rate / (1 + strength · (rate @ K_σ.t()))` in divisive mode; strength via softplus of the raw
  parameter when trainable.
- **Feedback evidence** — `predictive_feedback_evidence()`, lines 226–244:
  `posterior_prior_excess` = `relu(36·softmax(logits) − 1)`. At h≈0 (t0) the softmax is ~uniform
  ⇒ `f ≈ 0`: **feedback, and therefore `pred_inhib`, is essentially silent at the assay baseline.**
- **Config plumbing** — `model_config()` lines 283–307 and `build_tuned_from_config()` lines
  309–332: every `pred_inhib_*` / `local_comp_*` constant round-trips through the checkpoint dict.

### 2.2 The trainer — `/home/vishnu/neuroips_analysis/heatmap_sweep_20260818/harness/train_sweep.py` (944 lines; proven S2 path; itself a frozen root — coder must byte-copy)

- **`MODEL_CONFIG`** lines 38–59 — the only place the top-pick diff touches:
  `"pred_inhib_strength": 0.0` (line 45), `"pred_inhib_sigma_channels": 0.65` (line 46),
  `"local_comp_sigma_channels": 2.0` (line 54). No CLI overrides exist for these constants
  (verified `parse_args`, lines 759–846) — editing the copied file is the intended path.
- **Pretrain** — `run_pretrain()` lines 443–538: builds via `build_tuned_from_config(MODEL_CONFIG)`
  (line 448), task-only backward (line 495), `--pretrain-steps` default 3000 (line 786).
  ⇒ a fresh pretrain automatically contains the mechanism from step 1.
- **Arm** — `run_alpha()` lines 599–756: builds from `MODEL_CONFIG` (line 606), loads the common
  pretrain state with an identity-hash gate (lines 612–615; passes under a mechanism-config change
  because the kernels are non-persistent), objective
  `task_weight * losses["task"] + alpha * losses["energy"]` (line 688), `--axis-steps` default
  8000 (line 789), checkpoints every 250 steps (`--checkpoint-every`, default).
- **Checkpoint payload** — lines 401–438: stores `"tuned_net_config": MODEL_CONFIG` (via
  `model_config(net)`), plus `freeze_local_comp` (line 435) and `task_weight`.
- **Resume-metadata gate** — lines 637–645: steps/seed/alpha/task_weight/freeze/feedback-mode must
  match; run in a **fresh scratch run-dir** to avoid any stale-resume interaction.
- **Freeze flag** — `--freeze-local-comp` is `BooleanOptionalAction`, default True (lines 817–822)
  ⇒ `--no-freeze-local-comp` is available without code changes (used by candidate C only).
- **Mechanism observables** — `mechanism_statistics()` lines 553–562: logs softplus gains, the SOM
  margin `g1 − g2·g0`, effective k `g3 − g4·relu(margin)`, and the effective local-comp strength —
  already sufficient for the development gate's parameter-motion check.

### 2.3 The assay — frozen tools, config-driven rebuild (no assay edits needed)

- `/home/vishnu/neuroips_rnn_recreation_20260808/repo/tools/assay_emergent_task_energy_axis.py`
  lines 303–306 (`load_arm`): `net = tuned.build_tuned_from_config(checkpoint["tuned_net_config"])`
  then `net.load_state_dict(...)` ⇒ **any mechanism enabled in the trainer's MODEL_CONFIG is
  faithfully present at measurement time.** Lines 331–336: if the checkpoint declares
  `freeze_local_comp=True` the assay asserts `local_comp_strength_raw` equals the common pretrain
  tensor — satisfied by the top pick (freeze kept, strength untouched); inapplicable to the
  unfrozen candidate C (`freeze_local_comp=False` ⇒ check skipped).
- Profile convention — `/home/vishnu/neuroips_outputs/orientation_figs_20260819/make_orientation_figs.py`:
  `PLOT_OFFSETS = range(−12, 13)` (line 77), `CENTER_OFFSETS = (−1, 0, 1)` = ±5° (line 78),
  `FLANK_OFFSETS = (±3, ±4, ±5, ±6)` = ±15–30° (line 79), `FINAL_STEP = 4` (line 83);
  gray baseline = literal t0 first-stimulus response under the normal feedback-on unroll; color =
  continuation-A final step (lines 13–16, 134–141). `center_flank()` at line 95.
- **Assay-mechanism interaction (a decisive design fact):** the baseline state has `f ≈ 0`
  (uniform softmax at fresh hidden state ⇒ `relu(36/36 − 1) = 0`), while the continuation-A final
  step has a confident, center-concentrated `f`. A *feedback-driven* surround therefore subtracts
  from the flanks **only in the measured continuation state**, leaving the baseline untouched —
  maximal differential leverage on the flank/baseline ratio. A *rate-driven* mechanism
  (local_comp) acts in both states and largely cancels in the ratio (see §3, candidate C).

---

## 3. Ranked candidates

### RANK 1 (top pick) — feedback-recruited SOM-like broad subtractive surround (unifies protocol candidates a and b)

**Exact minimal diff** (in `train_flank.py`, a byte-copy of `harness/train_sweep.py` placed under
`/home/vishnu/scratch/flank_sharpening_20260819/`; two constants in `MODEL_CONFIG`, lines 45–46 of
the copy):

```python
    "pred_inhib_strength": 0.5,          # was 0.0
    "pred_inhib_sigma_channels": 4.0,    # was 0.65
```

Nothing else changes: no library edit, no loss edit, no CLI edit, no assay edit. Run command
(single-seed cheap-prove class, ~20 min):

```bash
cd /home/vishnu/scratch/flank_sharpening_20260819
PYTHONHASHSEED=0 python3 -B train_flank.py \
  --run-dir runs/predinhib_s0p5_sig4 --seed 8 --alphas 0.0 \
  --recurrent-cell rnn_tanh          # exact regime of the frozen α=0.0 network
```

(`sys.path` note for the coder: the copy imports `tools.tuned_emergence_lib` from the frozen repo
exactly as the original does; keep that import path untouched.)

**What the equation becomes.** With `K = local_circular_matrix(4.0)` (row-normalized) and s = 0.5,
line 168 of the lib reads, per channel i with predicted channel j:

  rate_i = relu( drive_i + [g3 − g4·som_gain]·f_i − 0.5·(K f)_i − … )

For a center-concentrated f this is a **difference-of-Gaussians on the feedback path**: the center
keeps the direct excitation `k·f_j` (k ≈ 0.55 at init, trained upward in the sharpening regime)
and loses only the self-pooled term `0.5·K_jj·f_j`; every non-predicted channel receives pure
subtraction `0.5·K_ij·f_j` with no offsetting excitation.

**Computed footprint numbers (N=36, row-normalized kernel):**

| σ (ch) | K(0) | K(3) | K(4) | K(5) | K(6) | 2·ΣK(3..6) = mass in flank band |
|---|---|---|---|---|---|---|
| 0.65 (shipped default) | 0.6135 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.000** |
| 2.0 | 0.1995 | 0.0648 | 0.0270 | 0.0088 | 0.0022 | 0.205 |
| **4.0 (chosen)** | **0.0997** | **0.0753** | **0.0605** | **0.0457** | **0.0324** | **0.428** |
| 5.0 | 0.0798 | 0.0667 | 0.0580 | 0.0484 | 0.0389 | 0.424 |

Two readings: (i) the shipped σ=0.65 default is *narrower than the excitation* — as shipped, this
hook is a center-attacking mechanism, i.e. the known failure mode; the σ change is what converts
it into a flank-targeting one. (ii) at σ=4.0, 42.8% of each unit of feedback-recruited inhibition
lands exactly in the ±15–30° assay flank band, while the center's self-inhibition coefficient is
s·K(0) ≈ 0.050 — only 7.2% of the initial direct gain g3 = 0.693 (and less after training raises
k). σ=4.0 ch = 20° also matches the measured biology (§5: Zhang 2014 note 37 — SOM-mediated
top-down surround suppression peaks at 200 μm ≈ 20° of visual angle; Adesnik 2012 — SOM spatial
footprint ~4× the pyramidal preferred size).

**Strength choice and pre-registered ladder.** s = 0.5 ≈ 0.9·k_init keeps the surround a moderate
fraction of the center drive, consistent with SOM silencing removing ~30% (not all) of suppression
in Adesnik 2012. Because the trained magnitude of f is regime-dependent (f can reach 36·p−1 ≫ 1
for confident softmax), the honest statement is that s sets a *ratio*, and the right absolute value
is an empirical single-seed question. Pre-registered ladder, one cheap retrain each, in order:
**s = 0.5 → 0.25 (if H or center degrade) → 1.0 (if flank effect < criterion but vitals clean)**.
σ stays fixed at 4.0 across the ladder (anatomy, not a tuning knob); the ladder is reasoned
mechanism-scaling, not parameter darts (§3 cheap-prove).

**Why this wins:** (1) smallest possible diff — two constants, all frozen tools keep working
by construction; (2) strongest biological grounding of the three (four primary sources bear
directly on it, §5); (3) it is the only candidate whose suppression is *feedback-gated*, firing
exactly in the assay's measured state and not in its baseline (§2.3); (4) it realizes protocol
candidates (a) and (b) simultaneously — the SOM-like broad pool *is* the inhibitory arm of the
Mexican hat, with the existing `g3·fb` as the excitatory arm.

### RANK 2 (candidate C) — broaden + unfreeze the existing divisive normalization pool

**Diff:** `"local_comp_sigma_channels": 2.0 → 4.0` (line 54 of the copy) and run the arm with
`--no-freeze-local-comp` (flag already exists) so `local_comp_strength_raw` trains in the arm.
Biology: Carandini & Heeger 2012 (normalization pool broader than the summation field is the
definitional requirement for surround suppression); Nurminen 2018 (divisive ROG model beats
subtractive DOG in 79% of feedback-inactivation cells).

**Why ranked below:** local competition is driven by the **current rates**, not by feedback, so it
is equally active at the t0 baseline and at the continuation step; the flank/own-baseline ratio —
the pass criterion — largely cancels the mechanism (§2.3). Its divisive form also rescales the
whole profile rather than differentially removing flank mass. Expected outcome: mild profile
narrowing, weak movement of the ratio criterion. Keep as the fall-back if RANK 1 fails *and* the
debugger's diagnosis points at subtractive-inhibition pathology (e.g. dead flank channels), since
divisive suppression cannot push rates to hard zero.
Note if adopted: `freeze_local_comp=False` is stored in the checkpoint, so the assay's
common-tensor equality check is skipped by design (assay lines 331–336) — still assayable.

### RANK 3 — per-channel k_i (literal protocol candidate a)

Replace the scalar gains with 36-vector `circ_raw` so each channel's SOM/VIP balance is learned.
Requires editing the frozen lib or shipping a subclass module (~50+ lines), changes `state_dict`
shapes, and silently breaks every frozen assay tool that rebuilds from config. Not minimal, and
its extra freedom is not needed for the goal: RANK 1 already delivers the broad-pool asymmetry
with the biology-fixed footprint. Hold in reserve only if the debugger proves the scalar motif
itself is the binding constraint.

---

## 4. Parameterization recommendation (learned vs bio-fixed)

- **Footprint σ = 4.0 channels — bio-FIXED.** Anatomy, not objective-shaped: SOM/feedback
  surround scale is an empirical constant (20°; Zhang note 37, Adesnik Fig. 1). Fixing it also
  removes any suspicion that a learned width was implicitly fit to the assay bands, and the
  row-normalized kernel makes σ a pure *shape* parameter (mass is conserved).
- **Strength s = 0.5 — bio-FIXED in the primary run** (ladder §3). The lib exposes no trainable
  path for `pred_inhib_strength` (plain float; verified constructor lines 55–128), so "learned
  strength" would demand lib changes — exactly what minimality forbids. The learned adaptation
  happens *around* the fixed mechanism instead: `circ_raw` (k rebalancing), `W_fb` (feedback
  sharpening), and the recurrent cell all remain trainable and let training set the *effective*
  operating point (e.g. raising g3 to protect the center, sharpening f to focus the excitation).
- **Init values:** everything else at the proven S2 values (§1 table); pretrain 3000 / arm 8000
  steps; seed 8; α = 0.0 only for Phase 2.

---

## 5. Primary sources (all read IN FULL this session via campus-IP/OA; local copies in `papers/`)

1. **Adesnik H, Bruns W, Taniguchi H, Huang ZJ, Scanziani M (2012).** *A neural circuit for
   spatial summation in visual cortex.* Nature 490:226–231. doi:10.1038/nature11526.
   [PMC3621107; local: `papers/adesnik2012.txt`]
   Claims used: SOM interneurons lack surround suppression (SI 0.09±0.06) and sum activity over a
   ~4× broader footprint than pyramidal cells (preferred size 86±3° vs 22±2°) because their main
   excitation is horizontal L2/3 axons (241±85% of PC-level excitation; only 17±5% from L4);
   silencing SOMs reduces pyramidal surround suppression by 30±10% (p=0.00022) and facilitates
   responses to larger-than-preferred stimuli by 74±19% **while leaving preferred-size responses
   unchanged (−7±7%, p>0.45)** — the center-sparing, surround-targeting incidence this design
   copies structurally.
2. **Zhang S, Xu M, Kamigaki T, Do JPH, Chang W-C, Jenvay S, Miyamichi K, Luo L, Dan Y (2014).**
   *Long-range and local circuits for top-down modulation of visual cortex processing.* Science
   345:660–665. doi:10.1126/science.1254126. [PMC5776147; local: `papers/zhang2014.txt`]
   Claims used: top-down feedback axons produce center facilitation (+0.17±0.02 at 0 μm, p=4e−16)
   plus surround suppression (−0.15±0.03 at 200 μm, p=4e−6) by recruiting local interneurons;
   SOM+ cells preferentially carry the surround arm (their silencing converts surround suppression
   to facilitation), VIP+ cells the center arm (localized disinhibition) — the same VIP/SOM motif
   the model already lumps into scalar k, now completed with the missing broad SOM arm; **note 37:
   200 μm ≈ 20° of visual angle** — the quantitative anchor for σ = 4 channels; their discussion
   explicitly proposes the same circuit operating "in stimulus feature space", licensing the
   orientation-domain transplant.
3. **Nurminen L, Merlin S, Bijanzadeh M, Federer F, Angelucci A (2018).** *Top-down feedback
   controls spatial summation and response amplitude in primate visual cortex.* Nature
   Communications 9:2281. doi:10.1038/s41467-018-04500-5. [OA CC-BY, PMC5995810; local:
   `papers/nurminen2018.xml/.txt`]
   Claims used: selectively inactivating V2→V1 feedback *reduces* responses to stimuli in the RF
   (−32.0±6.03%, p<1e−5) and *increases* responses in the proximal surround (+29.2±7.14%,
   p<0.001; SI 0.21→0.006) — i.e. intact feedback simultaneously boosts the center and suppresses
   the near surround, exactly the target profile; a recurrent model (Schwabe-type) reproduces the
   full effect set with one mechanism, **asymmetric inhibition**: higher-threshold/higher-gain,
   horizontally-pooled (SOM-like) inhibitory units driven by the same feedback that excites E
   cells — the computational abstraction implemented here as `g3·f − s·K f`.
4. **Ma W-p, Liu B-h, Li Y-t, Huang ZJ, Zhang LI, Tao HW (2010).** *Visual representations by
   cortical somatostatin inhibitory neurons — selective but with weak and delayed responses.*
   J Neurosci 30:14371–14379. doi:10.1523/JNEUROSCI.3248-10.2010. [PMC3001391; local:
   `papers/ma2010.txt`]
   Claims used: SOM cells are orientation-tuned as strongly as excitatory neurons (unlike untuned
   PV) — justifying an orientation-*tuned* circular-Gaussian surround kernel rather than uniform
   inhibition; their delayed (20–25 ms), facilitating, distal-dendrite-targeting responses mark
   them as gates of *later-arriving intracortical/feedback* input rather than feedforward drive —
   supporting attachment of the pooled inhibition to the feedback signal `fb_pos`, not to the L4
   `drive`.
5. **Ben-Yishai R, Bar-Or RL, Sompolinsky H (1995).** *Theory of orientation tuning in visual
   cortex.* PNAS 92:3844–3848. doi:10.1073/pnas.92.9.3844. [PMC42058; local:
   `papers/benyishai1995.pdf/_full.txt`]
   Claims used: the canonical ring model J(Δθ) = −J0 + J2·cos(2Δθ) — broad/uniform inhibition plus
   orientation-peaked excitation (net Mexican-hat) — generates sharp, contrast-invariant
   orientation tuning intrinsically once the modulated part is strong enough; establishes the
   50-year lineage for lateral-inhibition-in-orientation-space sharpening that RANK 1 instantiates
   on the feedback path.
6. **Carandini M, Heeger DJ (2012).** *Normalization as a canonical neural computation.* Nat Rev
   Neurosci 13:51–62. doi:10.1038/nrn3136. [PMC3273486; local: `papers/carandini2012.txt`]
   Claims used: "the normalization factor typically responds to a broader set of stimuli than the
   summation field" and surround suppression follows because "the suppressive field covers a
   larger region … than does the summation field" — the broader-pool-than-drive requirement shared
   by all candidates and definitional for candidate C; normalization improves decodability /
   winner-take-all readout of population codes — the functional reason the task objectives can
   *favor*, not merely tolerate, the mechanism. (Their honest caveat is noted: in V1 the divisive
   phenomena are not uniformly GABA_A-dependent; the direct causal circuit evidence for the
   surround arm is refs 1–3 above, and this review is cited for the computation, not a specific
   synapse.)

---

## 6. Predicted mechanism of emergence (why flank suppression develops under the ORIGINAL objectives)

The objectives are untouched: task = 0.5·next_ce/ln36 + 0.5·population_vector_loss/2, energy
(inactive at α=0). Nothing references the profile. The prediction is that training *exploits*
the fixed connectivity rather than neutralizing it, through three pressures:

1. **The center boost is already task-paid-for.** The established α=0 result (+18–23% center) shows
   the gradient rewards confident predictive feedback: sharper f → larger `g3·f` at the predicted
   channel → better next-step CE and tighter population vector. Under RANK 1, every unit of that
   same f now *also* delivers `s·K f` subtraction to the surround. Flank suppression rides in on
   the gradient pressure that already exists; to remove it, training would have to abandon the
   center boost it demonstrably wants.
2. **The readout prefers the suppressed profile.** With `population_vector` + normalize, flank mass
   is angular noise: subtracting symmetric flank rate tightens the resultant vector and lowers both
   task terms (Carandini–Heeger's decodability/WTA argument). So the mechanism's side effect is
   task-aligned, and gradients on `W_fb`/`circ_raw` should *sharpen* f and *raise* k rather than
   dampen the loop — the observable signature the development gate checks.
3. **The assay contrast is built in.** Baseline (t0, h fresh) has f ≈ 0 ⇒ no surround subtraction;
   continuation-A final step has strong center-concentrated f ⇒ full subtraction on flanks whose
   own feedforward drive at ±15–30° is modest (L4 σ=12° ⇒ ~2.4 ch). Net prediction:
   **flank/baseline < 1 immediately and deepening as f sharpens over training; center/baseline
   stays > 1 because the center retains k·f − s·K(0)·f ≈ (k − 0.05)·f.**

Neutralization risk (the falsifiable alternative): training could route around the surround by
collapsing f (uninformative feedback). That would show up at the development gate as k flat/failing
and the center boost vanishing — an unambiguous kill signature distinct from "mechanism too weak".

## 7. The known failure mode, head-on

Prior fact: any energy-type pressure suppresses the CENTER first, because cost-driven suppression
lands where the rate mass is — the predicted channel. RANK 1 is not cost-driven at any point:

- **Incidence is set by connectivity, not magnitude.** `pred_inhib` subtracts `s·(K f)_i` — its
  spatial profile is the fixed kernel row, independent of where the rate mass sits. With f
  concentrated at j: center keeps `k·f − s·K(0)·f` (k = 0.55 init vs s·K(0) ≈ 0.05, a 7% tax at
  init, smaller as k trains up); flanks receive `−s·K(3..6)·f` with **no** compensating excitation.
  42.8% of the subtraction lands in the assay flank band by construction.
- **The α=0 target regime has zero energy term** (objective line 688 with α=0), so the
  center-attacking pathway is absent from the training signal entirely.
- **The structural analogy is exact in the biology:** Adesnik 2012's SOM silencing left
  preferred-size responses untouched while releasing large-stimulus suppression — surround-targeted,
  center-sparing incidence is what a broad pool with a narrow direct drive *does*, in vivo as here.
- Residual center tax and its control: if the single-seed run shows center < 1.15×, the ladder
  step s→0.25 halves the tax to ~3.6% of g3-init while still placing >0.2 of kernel mass in the
  flank band. That is the pre-registered remedy, not a new mechanism.

## 8. Development gate (protocol criterion 4) and risk flags

Trainer counts steps, not epochs; "ep40–50" maps to the mid-arm checkpoint. **Pre-registered gate
at arm step 4000 of 8000** (`alpha_0p0_latest.pt` exists at every multiple of 250):

- **G1 — parameters leave init:** `circ_raw` distance from softplus-inverse init > 0 and moving
  (trainer logs `mechanism_statistics` — effective k trajectory); `W_fb` weight change vs pretrain
  nonzero. Under RANK 1 the *mechanism strength* is bio-fixed by design, so "mechanism develops"
  is measured as the trained parameters adapting AROUND it: k rising (or at least not collapsing)
  and f sharpening (feedback confidence up).
- **G2 — profile trend visible:** standard-convention profile from the step-4000 checkpoint shows
  flank/own-baseline < 1.00 and below the frozen α=0 reference's −2 to −3%; center/baseline > 1.
- **G3 — vitals:** H on trajectory ≥ ~0.9 at mid-arm (final criterion ≥ 0.95); mean rate M not
  collapsed (no dead-ring: fraction of exactly-zero L2/3 channels at continuation step below ~1/2).
- **Kill on the spot** (per protocol) if G1 shows the neutralization signature (k collapsing, f
  flattening) or G2 is flat with clean vitals at s = 1.0 (mechanism refuted in this architecture).
  Any FAIL → debugger single-variable diagnosis before the next candidate.

**H ≥ 0.95 risk flag:** direction of concern is over-suppression, not readout damage — flank
subtraction *tightens* the population vector (§6.2), so moderate s should help H, but a too-large
s can relu-silence flank channels and remove ring context the recurrent cell uses for next-step
prediction. Watch G3's dead-channel count; remedy is the pre-registered s-ladder (0.5→0.25).
Secondary risk: pretrain (task-only, 3000 steps) now also carries the mechanism — expected benign
(same DoG logic, and §3-mandated "trained into the network" from the start), but the coder should
confirm pretrain references (H/D/R at pretrain end) stay in family with the S2 pretrain values
logged in the run-start event; a large pretrain excursion is a G3-class early kill.

## 9. Implementation notes for the coder (traps discovered during mapping)

1. Byte-copy `harness/train_sweep.py` → `/home/vishnu/scratch/flank_sharpening_20260819/train_flank.py`;
   edit ONLY the two `MODEL_CONFIG` constants (§3). The harness dir itself is a frozen root — do
   not write there.
2. Keep the frozen-repo import path; do not copy or edit `tuned_emergence_lib.py`.
3. Fresh `--run-dir` under scratch (resume-metadata gate, trainer lines 637–645, will refuse mixed
   configs anyway — but a fresh dir avoids the failure mode entirely).
4. Run pretrain + arm in one invocation (the script does both); do NOT reuse the historical S2
   pretrain checkpoint — the mechanism must be present in pretrain too (§3 bake-in rule), and the
   fresh pretrain is part of the ~20-min budget (3000 + 8000 steps).
5. Kernels are non-persistent buffers: nothing about the mechanism lives in `state_dict`; the
   checkpoint's `tuned_net_config` is the single source of truth (assay rebuilds from it, §2.3).
6. Assay with the frozen standard convention (same offsets/bins as
   `make_orientation_figs.py:77–83`); compare each arm to ITS OWN t0 baseline (criteria are
   ratios to own baseline).
7. Envelope: dev3, scratch-only persistence, PYTHONHASHSEED=0, `python3 -B`, GPU per lead's
   dispatch; one checkpoint in memory at a time (del+gc) per the standing RAM rule.

## 10. Open questions for the lead

- The pre-registered s-ladder (0.5 → 0.25 / 1.0) is written as part of the cheap-prove loop the
  coder may traverse autonomously on a FAIL of the magnitude kind (criteria unmet, vitals clean,
  gate G1/G2 alive). Confirm that reading, or reserve ladder steps for post-diagnosis dispatch.
- Phase 2 scope check: single-seed 8, α = 0.0 arm only (the "trained sharpening regime" of the
  goal). The α = 0.5 dampening arm is untouched by this phase; a follow-up question is whether the
  mechanism also rescues the ε-regime center-notch, but that is out of the named goal.

---
*Researcher, 2026-08-19. All line numbers verified against the frozen files this session; papers
fetched OA-first (Europe PMC / PMC author manuscripts / Europe PMC PDF render) and read in full;
no source file edited; only this study directory written.*
