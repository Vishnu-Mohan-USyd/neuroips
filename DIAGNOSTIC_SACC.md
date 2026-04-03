# Diagnostic: s_acc Stuck at Chance During Stage 2

## Failure characterization

s_acc (36-way sensory readout accuracy) hovers at 8-12% throughout Stage 2 training (1100 steps observed). With 12 anchor orientations in the HMM, effective chance = 1/12 = 8.3%. Stage 1 gating achieved 100% decoder accuracy on the same L2/3 representation. The decoder is effectively non-functional in Stage 2.

---

## Hypotheses under investigation

1. Decoder from Stage 1 checkpoint is untrained (random init) — Status: **CONFIRMED (PRIMARY)**
2. Decoder gradients drowned by other losses — Status: **FALSIFIED**
3. Readout window captures wrong (previous) stimulus — Status: **CONFIRMED (SECONDARY)**
4. Decoder sees wrong data — Status: **CONFIRMED (consequence of #1 + #3)**
5. Feedback corrupts L2/3 tuning — Status: **NOT TESTED (fb_scale=0 for first 1000 steps, s_acc already stuck)**

---

## Experiments run

### Experiment 1: Decoder weights analysis

- **Command**: Compared checkpoint decoder weights to fresh random initialization
- **Expected if trained**: Large weights with diagonal structure (maps L2/3 channel i → logit i)
- **Expected if untrained**: Small weights in [-0.167, 0.167] (Kaiming uniform for 36 inputs), no structure
- **Actual result**:
  - Checkpoint W range: [-0.166, 0.166] — exactly Kaiming init bounds (1/√36 = 0.167)
  - Diagonal/off-diagonal ratio: 0.169 (random — no trained structure)
  - Decoder output entropy: 3.579 (max entropy = 3.584) — near-uniform
  - Decoder accuracy on perfectly tuned L2/3: **0/12 = 0%**
- **Verdict**: CONFIRMED — checkpoint decoder is untrained random weights

### Experiment 2: Code flow trace (the decoder bug)

- **Finding**: Two separate `CompositeLoss` instances exist:
  1. `stage1_sensory.py:116` — `loss_fn = CompositeLoss(...)` — LOCAL to `run_stage1()`, decoder trained to 100%, then **discarded** (never returned)
  2. `train.py:112` — `loss_fn = CompositeLoss(...)` — in `main()` scope, decoder **never trained**, saved to checkpoint at `train.py:131`
- Stage 2 receives `loss_fn` from `main()` (untrained decoder)
- The checkpoint's `decoder_state` is the untrained decoder from `main()`, not from `run_stage1()`
- **Verdict**: CONFIRMED — this is a bug in the training pipeline

### Experiment 3: L2/3 tuning verification (is the signal there?)

- **Command**: Presented 12 orientations through L4→PV→L2/3 at Stage 1 checkpoint
- **Result**: L2/3 argmax correctly tracks stimulus orientation for all 12 orientations
  - `ori=0°: argmax=0, ori=45°: argmax=9, ori=90°: argmax=18, ...`
- **Verdict**: L2/3 IS properly tuned. The representation is correct. Only the decoder is broken.

### Experiment 4: Readout window dynamics (carry-over from previous stimulus)

- **Command**: Simulated 50-presentation sequence, measured L2/3 argmax at every timestep within each presentation
- **Results** (accuracy = fraction of presentations where argmax matches current stimulus):

| Timestep (within ON) | Argmax accuracy | Note |
|---|---|---|
| Step 0 | 0.082 | Previous stimulus |
| Step 1 | 0.082 | Previous stimulus |
| Step 2 | 0.082 | Previous stimulus |
| Step 3 | 0.122 | Starting to switch |
| **Step 4** | **0.122** | **READOUT window start** |
| **Step 5** | **0.122** | |
| **Step 6** | **0.224** | Partially switched |
| **Step 7** | **0.531** | **READOUT window end** — first time >50% |

- Window average argmax accuracy: **16.3%** (3 of 4 window steps show previous orientation)
- **Verdict**: CONFIRMED — readout window is too early. L2/3 hasn't switched to current stimulus.

### Experiment 5: Decoder ceiling accuracy at different readout points

- **Command**: Trained fresh linear decoders on L2/3 at various readout points (2450 train, 500-test split)
- **Results**:

| Readout point | Raw argmax acc | Trained decoder acc |
|---|---|---|
| Window avg (steps 4-7) | 16.3% | **25.7%** |
| Step 7 only | 62.4% | **61.2%** |
| End of ISI (step 11) | 84.4% | **91.4%** |
| Stage 1 gating (20 steps from zeros) | 100% | **100%** |
| **Observed in Stage 2 log** | — | **~8%** |

- **Verdict**: Even a perfectly trained decoder caps at ~26% on the current readout window. The signal is fundamentally too weak at steps 4-7 for high-accuracy decoding.

### Experiment 6: Gradient analysis (H2 falsification)

- **Command**: Ran one full Stage 2 forward+backward pass, measured gradient norms per parameter group
- **Results**:
  - Decoder gradient norm: 0.030
  - V2 gradient norm: 0.050
  - Feedback gradient norm: 0.037
- **Verdict**: FALSIFIED — decoder gradients are comparable to V2/feedback. Not drowned by other losses.

---

## Confirmed root causes

### Root cause 1 (PRIMARY): Decoder checkpoint bug — untrained decoder saved and used

**The `run_stage1()` function creates its own local `CompositeLoss` (line 116 of `stage1_sensory.py`), trains its decoder to 100% accuracy, then returns without passing the trained decoder back. The caller (`train.py` line 112) has a separate `CompositeLoss` with an untrained decoder, which gets saved to the checkpoint (line 131) and passed to Stage 2.**

Evidence chain:
1. Checkpoint decoder weights are in [-0.166, 0.166] = Kaiming init bounds for nn.Linear(36, 36)
2. No diagonal structure (ratio = 0.17, indistinguishable from random)
3. Decoder output entropy = 3.579 / 3.584 = 99.9% of maximum (uniform)
4. 0% accuracy on perfectly tuned L2/3 input
5. Code trace confirms two separate `CompositeLoss` instances — trained one is discarded

### Root cause 2 (SECONDARY): Readout window captures previous stimulus, not current

**With `tau_l23=10` and readout at steps 4-7 of each 8-step presentation, L2/3 has only evolved through 0.4-0.7 time constants since onset. The carry-over from the previous stimulus dominates the activity pattern.**

Evidence chain:
1. Per-step argmax accuracy: steps 4-6 show ~12% (previous stimulus), step 7 shows 53%
2. Window average argmax accuracy: 16.3%
3. Ceiling accuracy with trained decoder: 25.7% (window avg) vs 61% (step 7 only) vs 91% (end of ISI)
4. Stage 1 used 20 steps from zeros → 100% accuracy. Stage 2 uses 8-step presentations with carry-over → L2/3 can't reset fast enough

---

## Secondary findings

1. **s_acc would still be low even with decoder fix alone**: The readout window limits a trained decoder to ~26%. Both bugs must be fixed for high s_acc.
2. **The sensory loss IS decreasing** (3.59 → 2.51 over 1100 steps): The random decoder IS slowly learning, but from an uninformative signal. The ~8% accuracy exactly matches exp(-2.51) ≈ 0.081 — the model is assigning ~8% probability to the correct class instead of 2.8%.
3. **Feedback is not the issue (for now)**: `fb_scale=0.0` for the first 1000 steps (burn-in), and s_acc is already stuck during this period. Feedback corruption (H5) may become relevant later but is not the current blocker.
4. **W_rec gradients are near-zero** (0.000009): gain_rec is frozen during burn-in as designed. This is expected.

---

## Alternative hypotheses ruled out

| Alternative hypothesis | Test | Result | Verdict |
|---|---|---|---|
| `_theta_to_channel` mapping bug | Tested all 12 anchors + edge cases | All correct | RULED OUT |
| `strict=False` hiding key mismatch | Tried `strict=True` | All keys match perfectly | RULED OUT |
| Loss and accuracy use different data | Code trace | Both use same `r_l23_windows` + `true_thetas` | RULED OUT |
| Decoder not in optimizer | Inspected param groups | Decoder in group 3, lr=0.001 | RULED OUT |
| Gradient detach/no_grad blocks decoder | Checked `requires_grad` + `grad_fn` | `r_l23_windows.requires_grad=True`, grad_fn exists | RULED OUT |
| Gradient clipping kills decoder grad | Checked clip target | Clip applies to `net.parameters()` only; decoder is in `loss_fn` | RULED OUT |
| Batch orientations too sparse | Counted distribution | 12 anchors + jitter covers 36 channels | RULED OUT |
| HMM repeats save most readouts | Counted orientation changes | 80% of presentations have >10° change (NEUT state) | RULED OUT (makes RC2 worse) |
| torch.compile changes L2/3 numerics | N/A for this analysis | Only affects speed, not computation | RULED OUT |

---

## Additional confirmation experiments

### Confirmation A: Fresh decoder CAN learn from this L2/3

- Trained a fresh nn.Linear(36,36) on Stage 1-style L2/3 (20 steps, zeros init) for 200 epochs
- **Result**: 80.6% accuracy (would reach ~100% with more epochs)
- Same L2/3 data with checkpoint decoder: **2.8%** (pure chance)
- **Proves**: The L2/3 representation is learnable. The checkpoint decoder is the sole problem.

### Confirmation B: L2/3 parameters are NOT corrupted

| Parameter | Checkpoint value | Expected (from prior Stage 1 analysis) | Match |
|---|---|---|---|
| `w_pv_l23.gain` | 0.1751 | ~0.175 | YES |
| `gain_rec` | 0.9644 | ~0.964 | YES |
| `pv.w_pv_l4` | 0.0968 | ~0.097 | YES |
| `pv.w_pv_l23` | 0.0716 | ~0.071 | YES |
| `sigma_rec` | 11.5037 | ~11.5 | YES |

Cross-checked against independent ablation checkpoint — values match to 4 decimal places. **Rules out** L2/3 parameter corruption.

### Confirmation C: Readout extraction code is correct

- Tested `extract_readout_data()` with synthetic data where values encode presentation index + timestep
- Window mean matched expected value to 5 decimal places for all presentations
- **Rules out** an indexing bug in the readout extraction

### Confirmation D: Gating checks FAIL with checkpoint decoder

- Re-ran `_run_gating_checks()` using checkpoint decoder: `decoder_accuracy_90: False` (5/5 non-decoder checks pass)
- Same network, same L2/3 — only the decoder differs
- **Proves**: The checkpoint's gating result `{'decoder_accuracy_90': True}` was produced by a *different decoder instance* (the one local to `run_stage1()`)

---

## Root cause locations

| Bug | File | Line | Description |
|---|---|---|---|
| Decoder not returned | `src/training/stage1_sensory.py` | 116 | `loss_fn = CompositeLoss(...)` is local, trained decoder discarded on return |
| Wrong decoder saved | `scripts/train.py` | 131 | `loss_fn.orientation_decoder.state_dict()` saves untrained decoder from line 112 |
| Wrong decoder to Stage 2 | `scripts/train.py` | 172 | `loss_fn` (untrained) passed to `run_stage2()` |
| Readout window too early | `src/training/stage2_feedback.py` | 126-131 | `window_start=4, window_end=7` with `tau_l23=10` — insufficient for switching |
