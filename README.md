# SOM/VIP sharpen ↔ dampen: expectation effects in a V1/V2-style cortical circuit

## Current validated workflow

> **Start here:** the current result is the four-seed
> [emergent task–energy axis](docs/emergent_task_energy_axis.md), not the older
> Phase A/Phase B experiments documented below. The current workflow trains six
> arms from one common task pretrain, assays fixed operational
> continuation/reversal pairs, and regenerates the tracked comparison figures.

| Entry point | Purpose |
| --- | --- |
| [`docs/emergent_task_energy_axis.md`](docs/emergent_task_energy_axis.md) | Canonical architecture, equations, protocol, four-seed results, limitations, and exact RTX 5090 reproduction |
| [`tools/tuned_emergence_lib.py`](tools/tuned_emergence_lib.py) | Current fixed orientation basis, L2/3 circuit, recurrent predictor, and feedback timing |
| [`tools/train_emergent_task_energy_axis.py`](tools/train_emergent_task_energy_axis.py) | Common pretrain and six task–energy alpha arms |
| [`tools/assay_emergent_task_energy_axis.py`](tools/assay_emergent_task_energy_axis.py) | Fixed 216-pair assay and decoding/rate/shape metrics |
| [`tools/plot_emergent_reference_figures.py`](tools/plot_emergent_reference_figures.py) | Four-seed checkpoint replay, literal first-stimulus comparator, JSON, and plots |
| [`figures/emergent_reference_comparison/`](figures/emergent_reference_comparison/) | Machine-readable aggregate data and the four current figures |

For this workflow, executable code and checkpoint contents are primary;
per-seed summaries/assays feed `plot_data.json`, which feeds the PNGs. The
plotter remeasures checkpoints but is not a phenotype pass gate.

This organization patch intentionally makes the validated six-alpha,
posterior-excess, frozen-local-competition protocol the trainer CLI default.
The numerical kernels and losses are unchanged for explicit validated
commands. Legacy behavior is selected explicitly with
`--feedback-mode baseline --no-freeze-local-comp --alphas 0.1 0.3 0.5 0.7 0.9`;
see the [canonical CLI note](docs/emergent_task_energy_axis.md#current-and-legacy-cli-defaults).

## Historical Phase A/Phase B lineage

The remainder of this README records the repository's earlier Phase A/Phase B
experiments. It is retained for provenance and is not the source of truth for
the current task–energy figures.

A small, from-scratch model of how *top-down expectation* reshapes a sensory representation. The whole
model is one short, pure-PyTorch file (`simple_net.py`) plus two self-contained experiment folders.

> **Reproduction.** Independently verified by an end-to-end reproduction from a fresh package-root
> invocation (no install, no `PYTHONPATH`): Phase A is **bit-identical**; Phase B shows the **4-seed**
> runtime switch, confirmed by the decisive `g_ctx`-lesion control (`|attend − save| = 0`). The release
> entry points `train_switch.py` and `proveout_switch.py` both reproduce — on the committed seed-1
> checkpoint **and** a fresh seed-23 retrain.

## 1. What this is

Cortex receives top-down predictions about what it is about to see. A long-standing question is whether
that feedback **sharpens** the representation of an *expected* stimulus (amplifies it, for precision) or
**dampens** it (suppresses the already-predicted stimulus, to save activity). Both are reported in the
biology, and they look contradictory.

This package builds a minimal V1/V2-like circuit — a fixed sensory layer, a trainable cortical layer, and
a recurrent predictor that feeds its prediction back down through a **Dale-compliant SOM/VIP inhibitory
microcircuit** — and shows that *sharpen and dampen are two settings of one circuit*, not two different
mechanisms. The result comes in two acts (Phase A → Phase B).

## 2. The A → B story

**Phase A — the regime *emerges*; nothing is hand-set.** One fixed SOM/VIP circuit is trained under a
single objective that trades *task accuracy* against a *metabolic energy cost* on cortical activity.
Sweeping that trade-off, the circuit **chooses for itself** whether feedback sharpens or dampens the
expected stimulus: low energy pressure → sharpen, high energy pressure → dampen, sliding smoothly between
them. No sign, no switch, and no regime is wired in — it is set entirely by the learned, non-negative
inhibitory gains.

**Phase B — *one* trained net switches at runtime, no retraining.** A single network learns *both*
regimes at once and flips between them on the fly, controlled only by the **sign of one context input**
(`+1` "attend" → sharpen, `−1` "save" → dampen). The same weights, the same circuit; only a context bit
changes.

**Critical framing (read this carefully).** Sharpen and dampen are an **energy-for-fidelity trade, not
"on vs off."**
- **Sharpen** = high-precision but metabolically costly: the expected channel is amplified several-fold.
- **Dampen** = energy-economical at the cost of lower L2/3 decoding fidelity: the already-predicted
  stimulus is encoded far more cheaply.
- **Dampen does *not* mean ignoring or discarding the input.** Both modes still represent the
  orientation — next-step prediction stays at ~80–82% either way. The dampened mode simply *spends less*
  to encode a stimulus it already expected.

## 3. Architecture (in words)

```
 input orientation θ
        │
   ┌────▼─────┐   L4 ring: FIXED circular-Gaussian population code
   │   L4     │   36 channels, 5°/channel.  No parameters.
   └────┬─────┘
        │  W_ff  (trainable, rectified)
   ┌────▼─────┐   L2/3 ring: the cortical representation.
   │  L2/3    │◄──────────────┐  top-down prediction fed back DOWN
   └────┬─────┘               │  through the SOM/VIP microcircuit
        │                     │
   ┌────▼─────┐   W_fb  ┌─────┴─────┐
   │   GRU    │────────►│ next-step │  predicts the NEXT orientation,
   │ (hidden  │         │ prediction│  feeds it back one step later
   │   = 64)  │         └───────────┘
   └──────────┘
```

The L2/3 feedback operator is a **Dale SOM/VIP microcircuit** (with `drive = W_ff(L4)` and `fb` = the
relu'd top-down prediction):

```
vip = relu(g_v·fb  [+ softplus(g_ctx)·ctx]   ← context term, Phase B only)
som = relu(g_s·fb − g_sv·vip)                 ← top-down drives SOM; VIP disinhibits it
r   = relu(drive + g_e·fb − g_ps·som)         ← L2/3 = feedforward + fb excitation − SOM inhibition
```

All five gains `g = softplus(·) ≥ 0`. **The minus signs are structural** — they encode cell types
(Dale's law), never negative weights. The regime is one emergent scalar of the gains: **VIP-dominant →
net excitation → sharpen; SOM-dominant → net inhibition → dampen.** Phase B adds a single context gain
`g_ctx` on the VIP cell; flipping the sign of `ctx` moves the circuit across the VIP/SOM balance.

## 4. Install

```bash
pip install -r requirements.txt
```

The model and assay code use PyTorch plus the Python standard library; the
current reference-figure entry point also uses Matplotlib. A **CPU-only** torch
build works (the device auto-falls back to CPU); a CUDA GPU is recommended for
training, while forward-only re-probes are cheap on CPU. Requires Python ≥ 3.10.

## 5. How to run

> **Confirmed by an end-to-end reproduction of the cleaned package:** every script below runs from the package
> root (or any neutral working directory) with **no `PYTHONPATH` and no install step** — each script
> bootstraps its own import of `simple_net.py` from a `__file__`-relative path. **CPU works; a GPU is
> optional** (it only speeds up the Phase-A grid and Phase-B training — all re-probes are cheap on CPU).
> Run from the package root:

**Phase A — emergent sharpen/dampen:**
```bash
python phaseA_somvip/grid_circuit.py            # 15-cell energy×task grid; regenerated ckpts -> grid_output/ (shipped ckpts untouched)
python phaseA_somvip/grid_circuit.py --out DIR  #   ...or --out DIR; point --out at phaseA_somvip to overwrite the shipped ckpts in place
python phaseA_somvip/mech_probe.py              # VIP/SOM causal knockouts on the saved ckpts
python phaseA_somvip/proveout_circuit.py        # single-instance prove-out (sharpen vs dampen)
python phaseA_somvip/reprobe_save_integrity.py  # reload ckpts from disk + re-probe (CPU-cheap)
python phaseA_somvip/smoke_circuit.py           # shape/regression smoke (uses ckpt_momentum.pt fixture)
```

**Phase B — runtime context switch:**
```bash
python phaseB_somvip/train_switch.py              # canonical generator: regenerates the seed-1 checkpoint in place
python phaseB_somvip/train_switch.py --out PATH   #   ...or pass --out PATH to write elsewhere (keep the shipped ckpt)
python phaseB_somvip/train_switch_seed0_gate.py   # seed-0 kill-gate
python phaseB_somvip/train_switch_robustness.py   # seeds 1–2 robustness
python phaseB_somvip/audit_switch.py              # independent audit + artifact battery (seeds 0 & 7)
python phaseB_somvip/proveout_switch.py           # pre-registered K1–K4 prove-out on the saved ckpt (CPU-ok)
python phaseB_somvip/reprobe_save_integrity_B.py  # reload checkpoint from disk + re-probe (CPU-cheap)
```

## 6. Expected results

All numbers are quoted from the two per-phase `RESULTS.md` (the source of truth), which read them
straight from the saved logs.

**Phase A — the regime tracks the energy/task ratio** (representative rows of the 15-cell grid;
`exp−floor > 0` = sharpen, `< 0` = dampen; "floor" = same net with feedback off):

| lam_energy | ce | ratio | net_fb_gain | regime | exp_r | floor | exp−floor | held % |
|----:|----:|------:|------------:|--------|------:|------:|----------:|------:|
| 0.00 | 1.0 | 0.000 | +0.495 | SHARPEN | 7.283 | 1.241 | **+6.04** | 81.8 |
| 0.05 | 1.0 | 0.050 | +0.077 | SHARPEN | 1.890 | 1.242 | +0.65 | 82.1 |
| 0.10 | 1.0 | 0.100 | −0.219 | DAMPEN | 0.185 | 1.231 | −1.05 | 81.9 |
| 0.20 | 1.0 | 0.200 | −0.575 | DAMPEN | 0.000 | 1.239 | −1.24 | 82.1 |

Across all 15 cells, `net_fb_gain` slides **monotonically +0.517 → −0.860**, crossing zero at the ratio
`lam/ce ≈ 0.05–0.10`, while held-out next-step prediction stays **~82% in every cell** — the regime flip
costs no accuracy. Sharpen amplifies the expected channel to **~6× the feedforward floor**; dampen
**suppresses it to 0.000** while the *unexpected* channel stays near floor (the suppression is
expectation-specific, not a global gain change). The saved checkpoints reproduce from disk: sharpen
**7.265 vs 1.229 (5.91×)**, dampen **0.000 vs 1.231**. Causal knockouts confirm the mechanism — in the
sharpener, removing VIP collapses the amplification (`r 7.27 → 2.00`); in the dampener, removing SOM
restores the channel (`r 0.00 → 4.43`).

**Phase B — one net, runtime switch, robust across seeds** (`ctx=+1` attend / `ctx=−1` save; same
weights):

| seed | attend `exp / floor / Δ` | save `exp / floor / Δ` | held att/save | g_ctx | switch |
|----:|--------------------------|------------------------|--------------:|------:|:------:|
| 0 | 2.251 / 1.239 / **+1.012** | 0.000 / 1.288 / **−1.288** | 81.0 / 80.6 | 1.916 | YES |
| 1 | 7.601 / 1.664 / **+5.938** | 0.543 / 1.695 / **−1.152** | 79.6 / 80.3 | 1.975 | YES |
| 2 | 6.950 / 1.749 / **+5.202** | 0.998 / 1.764 / **−0.766** | 80.3 / 80.0 | 1.815 | YES |
| 7 | 8.140 / 1.329 / **+6.812** | 0.840 / 1.358 / **−0.519** | 80.5 / 80.5 | 1.865 | YES |

Every seed switches the correct way, in one net at fixed weights; `g_ctx` **grew** from its 0.693 init to
~1.8–2.0 (the context drive is load-bearing). Two decisive controls: the feedback-off **floor is
context-independent** (`|Δ| = 0`), so the switch is not a baseline artifact; and **lesioning `g_ctx`
collapses the two contexts to identical** (`|attend − save| = 0`), so the switch is carried *entirely* by
the one trained context knob, not by shared-gain drift. Sign-agreement spans **5 distinct seeds**
(0, 1, 2, 7, 11) plus an independent audit re-test.

## 7. File manifest

| path | what |
|------|------|
| `simple_net.py` | the shared core — L4 ring, L2/3, GRU predictor, and the Dale SOM/VIP feedback operator; imported by both phases |
| `requirements.txt` | the single dependency (`torch>=2.0`) |
| `phaseA_somvip/` | **emergent regime.** `grid_circuit.py` (entry: 15-cell grid), `mech_probe.py`, `proveout_circuit.py`, `reprobe_save_integrity.py`, `smoke_circuit.py`, `mech_debug.py`; checkpoints `ckpt_circuit_sharpen.pt`, `ckpt_circuit_dampen.pt`, `ckpt_momentum.pt` (baseline fixture); `RESULTS.md` (full deep-dive) |
| `phaseB_somvip/` | **runtime switch.** `train_switch.py` (canonical generator that saves the checkpoint), `train_switch_seed0_gate.py`, `train_switch_robustness.py`, `audit_switch.py`, `proveout_switch.py`, `reprobe_save_integrity_B.py`; checkpoint `ckpt_ctxswitch_seed1.pt`; `RESULTS.md` (full deep-dive) |

**Checkpoint load conventions.** Phase-A checkpoints are plain `state_dict`s — load into
`SimpleNet(use_circuit=True)`. The **Phase-B checkpoint is a wrapper dict** `{'net','read','cfg'}` — load
with `ck = torch.load(...); net.load_state_dict(ck['net'])` into
`SimpleNet(use_circuit=True, context=True)`, **not** `load_state_dict` on the raw object (which silently
leaves the net at init). `reprobe_save_integrity_B.py` shows the correct load.

## 8. Caveats (honest scope)

- **Phase A** is a single seed on one frozen feedforward substrate. The monotone surface across 15 cells
  argues strongly against a fluke, but multi-seed robustness is the clean open follow-up. This is the
  first version where the sharpen/dampen **sign is *learned*, not hand-set.**
- **Phase B** is one substrate per seed. The switch **sign** is robust (4/4 seeds + audit), but the two
  arms are **not magnitude-symmetric** (attend +1.0…+6.8, save −0.5…−1.3) and the **neutral `ctx=0`
  baseline is off-center and seed-dependent** (e.g. seed 0 leans dampen, seed 7 leans sharpen). The ±1
  context knob reliably swings the regime *across* the floor in every seed; a centered, magnitude-
  symmetric switch is a further refinement, not part of the stated target.
- `simple_net.py` also contains **alternative feedback operators** — `signed_fb` push-pull, a
  `subtractive` Rao–Ballard predictive-coding variant — and alternative task regimes (`march`, `markov`).
  These are **exploratory lineage kept for the record; none are used by the Phase A/B headline results**,
  which all use the SOM/VIP circuit (`use_circuit=True`).

## 9. Full detail

Each phase's `RESULTS.md` is the authoritative, numbers-from-logs record, including the full grid, the
mechanism knockouts, the artifact battery, and the reproduction recipes:
- [`phaseA_somvip/RESULTS.md`](phaseA_somvip/RESULTS.md) — emergent sharpen/dampen from the objective.
- [`phaseB_somvip/RESULTS.md`](phaseB_somvip/RESULTS.md) — one net switching regime at runtime.
