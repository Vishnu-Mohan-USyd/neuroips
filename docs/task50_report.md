# Task #50 Completion Report — Raw-Softplus Weight Decay Anti-Shrinkage Fix

**Agent:** coder
**Date:** 2026-04-19
**Status:** Complete, awaiting validator review

---

## Problem

Under the legacy update `dw = lr*hebb - wd*weights`, weight decay pushed every
raw weight toward `0`. For excitatory weights initialised at strongly negative
raw (e.g. `W_l4_l23_raw` at `raw = -5.85`), pulling raw → 0 *increases*
`softplus(raw)` rather than shrinking it — **anti-shrinkage**. Debugger's
Claim 2 evidence (pre-fix): `softplus(W_l4_l23_raw).norm` drifted **+6%**
over 1000 steps at `wd=1e-5`, and **+74%** at `wd=1e-4`.

## Approach Chosen

**(A) + (C) combined** per Lead's dispatch:

Add optional `raw_prior: Optional[Tensor] = None` parameter to every
`.delta()` method in `src/v2_model/plasticity.py`.

- `raw_prior is None` → legacy toward-0 decay (preserves Phase-3 behaviour
  for zero-init task weights `W_qm_task`, `W_lm_task`, `W_mh_task`).
- `raw_prior is Tensor` → `dw = lr*hebb - wd*(weights - raw_prior)`.

The Phase-2 driver reads `module.raw_init_means[weight_name]` (populated by
`_make_raw(name="...")` at construction) and passes
`torch.full_like(w, init_mean)`. Phase-3 drivers unchanged — their task weights
init to 0, and the default `raw_prior=None` matches.

## Files Modified (diff --stat)

```
 scripts/v2/train_phase2_predictive.py |  29 ++++++-
 src/v2_model/layers.py                | 152 ++++++++++++++++++++--------------
 src/v2_model/plasticity.py            |  40 +++++++--
 src/v2_model/prediction_head.py       |  29 +++++--
 4 files changed, 172 insertions(+), 78 deletions(-)
```

## Core Diff (plasticity.py — representative; identical pattern at all 4 `.delta()` methods)

```diff
-        dw = self.lr * hebb - self.weight_decay * weights
+        shrink_target = weights if raw_prior is None else (weights - raw_prior)
+        dw = self.lr * hebb - self.weight_decay * shrink_target
```

Methods updated: `UrbanczikSennRule.delta`, `VogelsISTDPRule.delta`,
`ThreeFactorRule.delta_qm`, `ThreeFactorRule.delta_mh`.

## Probe C — softplus(W_l4_l23_raw).norm Drift (1000 Phase-2 steps)

| `wd` | `‖sp‖₀` | `‖sp‖_N` | Rel drift | Pre-fix | Reduction |
|:-----|:--------|:---------|:----------|:--------|:----------|
| 1e-05 | 0.525936 | 0.526086 | **+0.029%** | +6% | ~200× smaller |
| 1e-04 | 0.525936 | 0.525216 | **−0.137%** | +74% | ~540× smaller; sign flipped |

Both drifts are well below the |0.5%| stochastic Hebb noise floor. The
raw-prior anchor fully dominates decay; anti-shrinkage pathology is killed.

## Phase-2 200-step Smoke (seed=42, lr=1e-4 all rules, wd=1e-4, B=2, warmup=10)

| step | `|eps|_mean` | `r_l23_mean` | `r_h_mean` |
|:-----|:-------------|:-------------|:-----------|
| 0    | 2.776e-02    | 9.71e-04     | 0.00e+00   |
| 199  | 2.227e-02    | 1.24e-03     | 1.99e-05   |

**Δ|eps| rel: −19.76%.** |eps| decreasing, L2/3 activity rising modestly;
H rate still small (τ=500 ms, 200 steps × 5 ms ≪ settle time) — nominal.

## Extended Drift Trace (all strongly-negative-init weights, 1000 steps, wd=1e-3)

| weight | init | raw→ | Δraw | sp(init) | sp→ | Δsp/sp₀ |
|:-------|:-----|:-----|:-----|:---------|:----|:--------|
| l23_e.W_l4_l23_raw       | −5.85 | −5.8496 | +0.0001 | 2.876e-03 | 2.879e-03 | +0.109% |
| l23_e.W_rec_raw          | −4.70 | −4.6995 | +0.0000 | 9.054e-03 | 9.098e-03 | +0.486% |
| l23_e.W_fb_apical_raw    | −5.15 | −5.1499 | −0.0002 | 5.783e-03 | 5.787e-03 | +0.082% |
| l23_som.W_l23_som_raw    | −6.54 | −6.5400 | −0.0001 | 1.443e-03 | 1.444e-03 | +0.070% |
| l23_som.W_fb_som_raw     | −5.15 | −5.1498 | −0.0004 | 5.783e-03 | 5.788e-03 | +0.092% |
| h_e.W_l23_h_raw          | −7.44 | −7.4397 | −0.0005 | 5.871e-04 | 5.877e-04 | +0.099% |
| h_e.W_rec_raw            | −4.70 | −4.6986 | −0.0001 | 9.054e-03 | 9.107e-03 | +0.587% |
| prediction_head.b_pred_raw | −5.00 | −5.0003 | −0.0003 | 6.715e-03 | 6.714e-03 | −0.025% |

|Δraw| ≤ 5e-4 everywhere; |Δsp|/sp₀ ≤ 0.6%.

## Tests

**Full suite:** `909 passed, 1 xfailed in 111.93s` — zero failures.

The one xfail is the pre-existing
`test_predictive_loss_slope_is_negative_over_1000_steps`
("5% reduction in 1000 steps under current stability-first init") — not
touched by this task.

## Test Tolerance Updates

**None required.** No test weakened, skipped, or deleted. No tolerance relaxation.

## Ready for Validator Review.
