# Phase A — SOM/VIP microcircuit: sharpen vs dampen emerges from the objective

**Headline.** One fixed Dale SOM/VIP microcircuit, with no hand-set sign, self-selects between
*sharpening* and *dampening* the expected stimulus purely from the training objective (the ratio of
energy cost to task accuracy). The regime is set by learned non-negative gains, not by any
architectural switch. Prediction accuracy is unchanged (~82%) across the whole regime range.

This file + the scripts/checkpoints/logs beside it are the durable record. All numbers below are from
the saved logs (`grid_circuit.log`, `mech_probe.log`) and the save-integrity re-probe
(`reprobe_save_integrity.py`), not from memory.

---

## Architecture (the circuit, in `simple_net.py` `l23`, opt-in via `SimpleNet(use_circuit=True)`)

Fixed L4 ring (36 orientation channels, 5°/channel, circular-Gaussian) → trainable ReLU L2/3 ring
(`W_ff`) → GRU (hidden=64) that predicts the next orientation and feeds the prediction back into L2/3.
L2/3 is a Dale SOM/VIP microcircuit:

```
vip = relu(g_v * fb)                       # top-down feedback drives VIP
som = relu(g_s * fb - g_sv * vip)          # top-down drives SOM, VIP disinhibits SOM
r   = relu(W_ff(l4) + g_e * fb - g_ps*som) # Pyr: feedforward drive + fb excitation - SOM inhibition
```

All five gains `g = softplus(circ_raw) >= 0`. The minus signs are **structural** (cell-type / Dale's
law), never negative weights. Regime read as one scalar `net_fb_gain = g_e - g_ps*relu(g_s - g_sv*g_v)`
(>0 sharpen, <0 dampen). `fb` is the relu'd top-down prediction; "floor" = the same net with feedback
off (`r = relu(W_ff(l4))`).

---

## Result 1 — the regime is governed by the energy/task ratio (`grid_circuit.log`)

15-cell grid, `lam_energy ∈ {0, .02, .05, .1, .2}` × `ce_weight ∈ {.5, 1, 2}`; one shared frozen
seed-0 `phase1` substrate; each cell = 4000 steps of the energy+CE objective on momentum sequences.
`net_fb_gain` is **monotonic** and slides **+0.517 → −0.860** as energy pressure rises, crossing zero
at the **ratio `lam/ce ≈ 0.05–0.10` in all three `ce` rows** (the two axes collapse to the ratio —
metabolic-cost vs predictive-accuracy weighting). Held-out next-step prediction is **~82% in every
cell** — the regime flip costs no accuracy.

| lam | ce | ratio | net_fb_gain | regime | exp_r | unexp_r | floor | exp−floor | held% |
|----:|---:|------:|------------:|--------|------:|--------:|------:|----------:|------:|
| 0.00 | 0.5 | 0.000 | +0.517 | SHARPEN | 7.788 | 1.339 | 1.254 | +6.534 | 81.8 |
| 0.02 | 0.5 | 0.040 | +0.152 | SHARPEN | 2.654 | 1.313 | 1.238 | +1.415 | 82.1 |
| 0.05 | 0.5 | 0.100 | −0.245 | DAMPEN | 0.098 | 1.146 | 1.258 | −1.160 | 82.0 |
| 0.10 | 0.5 | 0.200 | −0.541 | DAMPEN | 0.000 | 1.050 | 1.247 | −1.247 | 81.9 |
| 0.20 | 0.5 | 0.400 | −0.860 | DAMPEN | 0.000 | 0.884 | 1.264 | −1.264 | 81.9 |
| 0.00 | 1.0 | 0.000 | +0.495 | SHARPEN | 7.283 | 1.337 | 1.241 | +6.042 | 81.8 |
| 0.02 | 1.0 | 0.020 | +0.299 | SHARPEN | 4.456 | 1.301 | 1.211 | +3.245 | 82.0 |
| 0.05 | 1.0 | 0.050 | +0.077 | SHARPEN | 1.890 | 1.274 | 1.242 | +0.648 | 82.1 |
| 0.10 | 1.0 | 0.100 | −0.219 | DAMPEN | 0.185 | 1.186 | 1.231 | −1.046 | 81.9 |
| 0.20 | 1.0 | 0.200 | −0.575 | DAMPEN | 0.000 | 1.090 | 1.239 | −1.239 | 82.1 |
| 0.00 | 2.0 | 0.000 | +0.512 | SHARPEN | 7.464 | 1.340 | 1.215 | +6.249 | 81.7 |
| 0.02 | 2.0 | 0.010 | +0.416 | SHARPEN | 6.019 | 1.259 | 1.237 | +4.782 | 82.0 |
| 0.05 | 2.0 | 0.025 | +0.276 | SHARPEN | 4.095 | 1.317 | 1.223 | +2.872 | 81.9 |
| 0.10 | 2.0 | 0.050 | +0.096 | SHARPEN | 2.054 | 1.305 | 1.240 | +0.815 | 82.1 |
| 0.20 | 2.0 | 0.100 | −0.249 | DAMPEN | 0.092 | 1.177 | 1.239 | −1.148 | 82.2 |

Sharpen extreme: expected channel amplified to **~6× the feedforward floor** (7.79 vs 1.25). Dampen
extreme: expected **suppressed to 0.000** while the unexpected channel stays near floor. The learned
gains show the mechanism plainly — sharpen cells: `g_v≈g_sv≈0.79`, `g_s≈0.61`, `g_ps≈0.63`; dampen
cells: `g_v` falls to ~0.41–0.49 while `g_s` and `g_ps` both climb past ~1.1.

## Result 2 — the mechanism is causally VIP-disinhibition vs SOM-inhibition (`mech_probe.log`)

Direct activity readout + single-variable knockouts on the two saved checkpoints. The instrumented
forward is verified identical to `forward_seq` (`max|diff| = 0.00e+00`). At the predicted channel,
feedback on:

| checkpoint | VIP | SOM | r | kill VIP → | kill SOM → |
|------------|----:|----:|---:|------------|------------|
| sharpen | 11.44 | 0.00 | 7.27 | SOM 0→6.75, **r 7.27→2.00** | r unchanged (7.27→7.29) |
| dampen | 6.05 | 12.42 | 0.00 | SOM 12.42→14.14, r stays 0 | **r 0.00→4.43** |

So in the **sharpener**, VIP holds SOM at zero and feedback is pure excitation — remove VIP and SOM
springs back and the amplification collapses. In the **dampener**, SOM dominates and silences the
channel — remove SOM and the channel returns. The regime is *carried* by the VIP/SOM balance, not
merely correlated with the gains.

## Result 3 — save-integrity: the saved checkpoints still reproduce (`reprobe_save_integrity.py`)

Reloaded from disk (2026-06-24): `ckpt_circuit_sharpen.pt` → expected `7.265` vs floor `1.229`
(**5.91×, SHARPEN**); `ckpt_circuit_dampen.pt` → expected `0.000` vs floor `1.231` (**0×, DAMPEN**).
The files on disk embody the result.

---

## Reproduction recipe

`SimpleNet(use_circuit=True)`, `torch.manual_seed(0)`, `phase1(net, steps=2000)` (trains `W_ff` +
decoder, feedback off — this frozen substrate is shared across all grid cells), then the energy+CE
objective for 4000 steps per cell (trains `gru` + `W_fb` + `circ_raw`; `W_ff`+decoder frozen) on
momentum sequences (`p_stay=0.9`, `S=12`, `batch=128`, `lr=1e-3`), with `lam_energy`/`ce_weight` set
per cell. The exact, authoritative driver is **`grid_circuit.py`** (grid), **`mech_probe.py`**
(mechanism knockouts), and **`proveout_circuit.py`** (single-instance prove-out). `mech_debug.py` is
the drive-vs-floor red-herring check (the pre-relu signed `drive≈0.09` at the orientation-index
channel is not the bump peak; the `~1.24` floor is post-relu `relu(drive)` — different quantities,
no bug).

## Manifest (all in this directory)

| file | what | md5 |
|------|------|-----|
| `ckpt_circuit_sharpen.pt` | trained sharpener (lam=0, ce=1) | `a082f790dccc6af357d967df6f001a2a` |
| `ckpt_circuit_dampen.pt` | trained dampener (lam=0.2, ce=1) | `2f5176565b32ef66785084d19937761b` |
| `ckpt_momentum.pt` | additive-feedback baseline (`use_circuit=False`); fixture loaded by `smoke_circuit.py` (regression) + `mech_debug.py` (additive arm) | `878cdf7eab724c1974bd8853434cee1f` |
| `grid_circuit.py` / `grid_circuit.log` | 15-cell energy×task grid + raw output | — |
| `mech_probe.py` / `mech_probe.log` | mechanism knockouts + raw output | — |
| `proveout_circuit.py` / `proveout_circuit.log` | single-instance prove-out + raw output | — |
| `smoke_circuit.py` | backward-compat / shape smoke | — |
| `mech_debug.py` | drive-vs-floor red-herring resolution | — |
| `reprobe_save_integrity.py` | reload-and-reprobe save check | — |

## Caveats

Single seed / one frozen substrate. The monotone surface across 15 cells argues against a fluke, but
seed-robustness is the clean open follow-up. This is the first version where the **sign is learned**,
not hand-set — contrast the earlier hand-signed and reverted two-population error-neuron work.
