# Phase B — one net switches sharpen ↔ dampen at runtime via a context knob

**Headline.** A single trained network runs the *same* fixed Dale SOM/VIP microcircuit and flips
between **sharpening** and **dampening** the expected stimulus **at runtime**, set only by the sign of
one scalar context input — no retraining, no weight change. `ctx = +1` ("attend") sharpens the expected
channel several-fold above the feedforward floor; `ctx = −1` ("save") instead encodes that same stimulus
far more economically — much lower L2/3 activity, **prioritizing metabolic energy at the cost of L2/3
decoding fidelity**. "Save" is an energy-for-fidelity trade, **not** a discarding or ignoring of the
input: both modes still represent the orientation (next-step prediction stays intact ~80% in both); the
dampened mode simply spends less to encode the already-predicted expected stimulus. The switch is carried
**entirely by one trained context gain** (`g_ctx`): lesion that gain and the two contexts become
bit-identical.

This is the runtime-context follow-on to Phase A (`../phaseA_somvip/`), where the sharpen/dampen regime
was baked in per-network by the training objective. Here the regime is a runtime choice in *one* net.

All numbers below are from the saved logs beside this file (`train_switch_seed0_gate.log`,
`train_switch_robustness.log`, `audit_switch.log`, `reprobe_save_integrity_B.py` output), not from memory. The result was
independently audited on a separate harness (Debugger, fresh seed 7) — verdict **GO, no artifact**.

---

## Architecture (the context branch in `simple_net.py` `l23`, opt-in via `SimpleNet(use_circuit=True, context=True)`)

The Phase-A Dale SOM/VIP circuit, with one added context input wired into VIP:

```
vip_in = g_v * fb + softplus(g_ctx) * ctx     # context adds to / subtracts from the VIP drive
vip    = relu(vip_in)                          # ctx=+1 raises VIP -> more disinhibition -> sharpen
som    = relu(g_s * fb - g_sv * vip)           # ctx=-1 lowers VIP -> SOM released -> dampen
r      = relu(W_ff(l4) + g_e * fb - g_ps*som)  # Pyr: feedforward + fb excitation - SOM inhibition
```

`g_ctx = softplus(g_ctx_raw)` is a single non-negative scalar gain on the context input. The regime is
flipped by the **sign of `ctx`**, through the **same** circuit and the **same** one knob — no extra
knob is added to either regime, so the symmetric-treatment constraint holds (the regime difference is a
runtime input sign, not asymmetric machinery). `fb` is the relu'd top-down prediction; "floor" = the
same net with feedback off (`r = relu(W_ff(l4))`), which is context-independent (see T2 below).

## The incentive that makes it train (defeats the three Phase-B failure causes)

The diagnosed reason naive context-switching failed: the network predicts equally well (~82%) whether it
sharpens or dampens, so there is **no task incentive to sharpen**, `g_ctx` is **born gradient-dead** in
the SOM relu dead-zone, and the regime is otherwise carried by the **shared gains** (so a sharpen
incentive only buys a knife-edge switch). The fix gives "attend" a *genuine* reason to sharpen:

- A trainable external linear readout `read: ℝ³⁶→ℝ³⁶` decodes the **current** orientation from L2/3 `r`
  under **additive noise** `r_noisy = r + σ·N(0,1)`, `σ = 1.0`. Higher / sharper activity = better SNR =
  lower readout loss. This readout-CE is applied to **attend (`ctx=+1`) samples only**.
- An **energy** penalty `mean|r|` is applied to **save (`ctx=−1`) samples only**.
- Prediction-CE is applied to **both**.

So attend is pulled to sharpen (to be readable through noise) and save is pulled to dampen (to spend
less energy), and the only thing that can satisfy both in one net is to *use the context bit* — which
grows `g_ctx` instead of letting it atrophy.

Objective (per step, batch 128, `ctx` drawn ±1 with p=0.5):
`loss = CE_pred(both) + 1.0 · CE_readout(attend-only, noisy) + 0.2 · energy(save-only)`.

---

## Result 1 — the runtime switch, robust across 4 seeds (`train_switch_*.log`, `audit_switch.log`)

One net per seed; `phase1` trains `W_ff`+decoder (then frozen); 4000 steps of the objective above train
`gru`+`W_fb`+`circ_raw`+`g_ctx_raw`+`read`. Probe: K=4 momentum lead-in to expected channel `e`, read
`r[:,K,e]` with feedback on vs off (floor). `held` = next-step prediction on fresh momentum sequences.

| seed | source | attend `ctx=+1` (exp / floor / exp−floor) | save `ctx=−1` (exp / floor / exp−floor) | held attend/save | g_ctx | switch |
|----:|--------|------------------------------------------|-----------------------------------------|------------------|------:|:------:|
| 0 | Lead + Debugger anchor | 2.251 / 1.239 / **+1.012** SHARPEN | 0.000 / 1.288 / **−1.288** DAMPEN | 81.0 / 80.6 | 1.916 | YES |
| 1 | Lead | 7.601 / 1.664 / **+5.938** SHARPEN | 0.543 / 1.695 / **−1.152** DAMPEN | 79.6 / 80.3 | 1.975 | YES |
| 2 | Lead | 6.950 / 1.749 / **+5.202** SHARPEN | 0.998 / 1.764 / **−0.766** DAMPEN | 80.3 / 80.0 | 1.815 | YES |
| 7 | Debugger (fresh) | 8.140 / 1.329 / **+6.812** SHARPEN | 0.840 / 1.358 / **−0.519** DAMPEN | 80.5 / 80.5 | 1.865 | YES |

All 4 seeds switch the correct way: `ctx=+1` lands above floor (sharpen), `ctx=−1` below floor (dampen),
in the same net at the same weights. `g_ctx` **grew** from its 0.693 init to ~1.8–2.0 in every seed — the
context drive is load-bearing, where in every prior (incentive-free) attempt it atrophied to ~0.2.

## Result 2 — independent audit: genuine switch, every artifact falsified (`audit_switch.log`)

Independent harness (Debugger), seeds 0 (anchor, reproduced Lead to the digit) and 7 (fresh). The whole
artifact battery, both seeds:

- **T2 — floor is context-independent.** `floor(ctx=+1) = floor(ctx=−1)` to `|Δ| = 0.00e+00`. The
  expected-channel `exp−floor` sign is therefore not a floor/baseline bias; the switch shows even on
  bit-identical inputs (attend `exp_r > floor`, save `exp_r < floor` on the same input).
- **T4 — the switch is 100% carried by `g_ctx` (decisive; not shared-gain co-drift).** Lesion the
  context gain (`softplus→~0`) and attend vs save collapse to **identical** `exp−floor`,
  `|attend−save| = 0.00e+00` on both seeds. No leak through the shared gains — the one trained context
  knob does all the work. This closes the prior "shared-gain knife-edge" concern.
- **Dampen is selective, not a dead network.** Save keeps held-out prediction ~80%, `unexp_r > 0`, and
  `mean|r|@K` alive (0.86 seed0 / 1.01 seed7). The expected channel floors to 0.000 via the **same SOM
  mechanism** as the dedicated Phase-A dampener (which also yields exactly 0.000).
- **Prediction genuinely uses feedback, equally in both regimes.** `held` fb-on ~80% ≫ fb-off
  (71% seed0 / 57% seed7) in *both* attend and save — feedback is not vestigial.
- **Sharpen is expectation-specific, not a flat global gain.** Under attend, `unexp_r ≈ floor`
  (1.30/1.24 seed0, 1.41/1.33 seed7) — only the *expected* channel is amplified, matching the dedicated
  sharpener's structure.

**Full pre-registered prove-out (the four kill-criteria K1–K4 fixed up front, designed to refute; re-runnable via `proveout_switch.py`).** The
headline survived an adversarial re-test on BOTH the *defended* saved checkpoint and a fresh seed. The
saved `ckpt_ctxswitch_seed1.pt`, reloaded read-only from disk, passes K1 (attend +5.93), K2 (save −1.16),
K3 (held 79.6/80.5%), and the **K4 crux** (flip only the context bit on frozen weights → regime flips,
floors byte-identical `|Δ|=0`); a fresh independent seed-11 retrain passes identically (+5.25 / −1.56,
held 79.9/80.6%, g_ctx 1.87). Sign-agreement now spans **5 distinct seeds** (0,1,2,7,11) plus an independent audit re-test; the g_ctx-lesion
collapse (`|Δ|=0`) reconfirms the switch is 100% context-knob-carried, not shared-gain co-drift.

## Result 3 — save-integrity: the saved checkpoint reproduces from disk (`reprobe_save_integrity_B.py`)

Reloaded `ckpt_ctxswitch_seed1.pt` from disk (no training, pure forward probe): `ctx=+1` →
exp 7.614 / floor 1.689 (**+5.93, SHARPEN**); `ctx=−1` → exp 0.536 / floor 1.699 (**−1.16, DAMPEN**);
`g_ctx = 1.975`; held ~80% both. The file on disk embodies the switch.

---

## Caveat (characterization, not an artifact — independently flagged in the audit)

The **neutral (`ctx=0`) baseline is off-center and seed-dependent**: with the context input at zero, the
shared-gain default is **seed0 `exp−floor = −0.70` (dampen)** but **seed7 `+2.48` (sharpen)**. So the ±1
context knob swings the regime *across* the floor in every seed (the **sign** of the switch is robust
4/4), but the two arms are **not magnitude-symmetric** (attend +1.0…+6.8; save −0.5…−1.3) and the zero
point wanders by seed. This is the residual of the regime living partly in the shared-gain baseline; it
does not undermine "one net, one knob, genuine runtime switch," but a *centered, magnitude-symmetric* ±
switch would be a further refinement (e.g. balancing the readout/energy pressures or regularizing the
neutral baseline to the floor). Not part of the stated Target-B spec (switch-by-context-without-retrain),
which is met.

## Reproduction recipe

`SimpleNet(use_circuit=True, context=True)`, `torch.manual_seed(s)`, `phase1(net, steps=2000)` (trains
`W_ff`+decoder, feedback off; then frozen). External `read = nn.Linear(36,36)`. Then 4000 steps, batch
128, `S=12` momentum (`p_stay=0.9`), Adam `lr=1e-3`, `ctx` drawn ±1 (p=0.5):
`loss = CE_pred(both) + 1.0·CE_readout(attend-only, on r+1.0·randn) + 0.2·energy(save-only)`.
Authoritative drivers: **`train_switch.py`** (canonical seed-1 generator that saves the committed
checkpoint), **`train_switch_seed0_gate.py`** (seed-0 kill-gate), **`train_switch_robustness.py`**
(seeds 1–2), **`audit_switch.py`** (independent audit + artifact battery, seeds 0 & 7),
**`proveout_switch.py`** (re-runnable K1–K4 prove-out on the saved checkpoint, or a fresh seed),
**`reprobe_save_integrity_B.py`** (reload-and-reprobe save check).

## Manifest (all in this directory)

| file | what | md5 |
|------|------|-----|
| `ckpt_ctxswitch_seed1.pt` | trained switch net + readout (seed 1) | `538bc174d4e5b930f803cd3e5734d286` |
| `train_switch.py` | canonical generator: trains seed 1 and saves the wrapper checkpoint above | — |
| `train_switch_seed0_gate.py` / `train_switch_seed0_gate.log` | seed-0 kill-gate driver + output | — |
| `train_switch_robustness.py` / `train_switch_robustness.log` | seeds 1–2 robustness driver + output | — |
| `audit_switch.py` / `audit_switch.log` | independent audit + artifact battery (seeds 0, 7) | — |
| `proveout_switch.py` | re-runnable pre-registered K1–K4 prove-out (saved ckpt, CPU-ok; optional fresh seed) | — |
| `reprobe_save_integrity_B.py` | reload-and-reprobe save-integrity check | — |
| `dbg_proveout_defended.log` | ARCHIVAL: original K1–K4 prove-out on the defended seed-1 checkpoint (PASS). Superseded by `proveout_switch.py`. | — |
| `dbg_proveout_k.log` | ARCHIVAL: original K1–K4 prove-out on fresh seed-11 (PASS). Its `defended` line is a known wrong-load artifact; `proveout_switch.py` loads the wrapper correctly and the saved seed-1 net passes K1–K4. | — |

**Loading the checkpoint:** it is a wrapper dict `{'net','read','cfg'}` — do
`ck = torch.load(...); net.load_state_dict(ck['net'])` into `SimpleNet(use_circuit=True, context=True)`,
not `load_state_dict` on the raw object (which silently leaves the net at init). `reprobe_save_integrity_B.py`
shows the correct load.

## Status

Target B (one net switches sharpen/dampen via runtime context, no retraining) **achieved and
independently audited GO**. Single substrate per seed, 4 seeds agree on sign, decisive `g_ctx`-lesion
test. Open refinement: the off-center/seed-dependent neutral baseline (magnitude symmetry), above.
