# Stage-1 ctx_pred attempt #1 / #2 — diagnostic snapshot

Checkpoint: `expectation_snn/data/checkpoints/stage_1_ctx_pred_seed42.npz`.

Attempts #1 (commit 5317540) and #2 (commit 31e6e98) produced
bit-identical weights and firing rates — see commit 31e6e98 for the
`_stage1_h_cfg(h_cfg)` override discovery. This snapshot is against
the shared final state, which represents the network at
effective `inh_w_max = 1.5` (the `_stage1_h_cfg` hardcoded value).

## Root-cause mechanism: init × row-cap collapse (confirmed from checkpoint)

The `W_ctx_pred_final` mean is **exactly** `3.0 / 192 = 0.015625` (the `w_row_max / n_pre`
ratio) — not an artefact, a fixed point of the row-cap rescale.

**Evidence chain** (all from the checkpoint, no re-run):

1. Row-cap equation: `apply_modulatory_update` (`h_context_prediction.py:505–515`)
   rescales every row whose sum exceeds `w_row_max = 3.0` via
   `scales = w_row_max / row_sum`. After the rescale every capped row has
   `sum_j w[i, j] = 3.0` exactly, so per-synapse mean in that row = `3.0 / 192`.

2. Init row sums are already over the cap.
   - `W_ctx_pred_init ~ uniform(0, 0.05)` → init mean = 0.02502
   - Rows × synapses/row = 192 × 192 = 36864 ✓
   - Init row sum ≈ 0.02502 × 192 = **4.80**, but `w_row_max = 3.0`.
   - → Every row is over the cap at init; the *very first* gate's LTP increment
     pushes `w_new` higher, the cap fires on every row, and the row-sum is
     clamped to 3.0 uniformly. Per-synapse mean collapses to `3.0/192 = 0.015625`.

3. `gate_n_capped` is **constant 192 across all 360 gates** (min=max=192). That is
   the row count, not the synapse count (`h_context_prediction.py:515`:
   `n_capped = int(over.sum())` where `over` is per-row). All 192 rows hit the
   cap every single gate.

4. `gate_w_before` trajectory confirms the collapse happens within the first
   ~36 trials: bucket 1 (trials 0–35) min=0.01562, max=0.02502 (still has
   pre-collapse state); bucket 2+ (trials 36–359): min = mean = max = 0.01562.

5. Positive-feedback lock-in: once W_ctx_pred is floored at 0.015625, H_pred
   post-leader firing is weak → pre/post coincidence is weak → `gate_elig_max`
   decays **6×** across the schedule (bucket 1 mean 897 → bucket 10 mean 144).
   By late trials, elig is too small to fight the decay term, and the rule
   cannot recover even if the cap were lifted.

6. The forecast-gate confusion (pred_argmax == leader 98.9%, == expected_trailer
   0.0%) is the *structural signature* of an amplifier-not-predictor: with
   W_ctx_pred locked at uniform floor 0.015625, the ctx→pred mapping is
   effectively a flat projection; the pred ring's own recurrent dynamics dominate
   and just echo whatever ctx is currently bumping (= the leader).

**This is not a `inh_w_max` problem.** Neither attempt #1 (`inh_w_max=1.5`, forced
by `_stage1_h_cfg`) nor attempt #2 (`HRingConfig(inh_w_max=2.0)` silently
overridden by `_stage1_h_cfg`) could have passed: the W_ctx_pred collapse
happens in the first few gates, before Vogels has a chance to shape inhibition.

**Implication for the fix-parameter search**: the knob is `init × row-cap`
compatibility. Options:

- **A. Shrink init** (cleanest biological fix): `w_init_frac = 3.0/192/0.025 ≈ 0.02` would put init row sums
  at exactly `w_row_max`. Safer: `w_init_frac = 0.015` → init mean 0.0075, init
  row sum 1.44, leaves headroom for LTP to carry it up.
- **B. Raise cap**: `w_row_max = 5.0` or higher would admit the current init.
  Picks the bump-supporting weight range, but leaves the "every row over cap
  every gate" asymmetry in place.
- **C. Combined**: shrink init AND raise cap modestly. Most robust.

Debugger can now test each directly.

## Gate verdict

| check | value | band | result |
|---|---|---|---|
| h_bump_persistence_ms | 10.0 | [200, 500] | **FAIL** |
| h_preprobe_forecast_prob | 0.000 | [0.25, 1.00] | **FAIL** |
| no_runaway | 21.27 Hz | [0, 80] | PASS |

## Weight-distribution snapshot

### `ctx_ee_w_final`  (n=36672)

```
  n=36672  min=0.0011175  p01=0.0020689  p10=0.012538  p25=0.024722  median=0.049804  mean=0.08377  p75=0.076602  p90=0.15005  p99=0.54389  max=0.57386  std=0.11838
```
Histogram (text):

```
  [  0.001118,    0.02975)    11008  #######################################
  [   0.02975,    0.05839)    11264  ########################################
  [   0.05839,    0.08703)     6912  #########################
  [   0.08703,     0.1157)     2560  #########
  [    0.1157,     0.1443)      768  ###
  [    0.1443,     0.1729)     1024  ####
  [    0.1729,     0.2016)        0  
  [    0.2016,     0.2302)      256  #
  [    0.2302,     0.2589)        0  
  [    0.2589,     0.2875)        0  
  [    0.2875,     0.3161)        0  
  [    0.3161,     0.3448)        0  
  [    0.3448,     0.3734)        0  
  [    0.3734,      0.402)        0  
  [     0.402,     0.4307)      480  ##
  [    0.4307,     0.4593)     1680  ######
  [    0.4593,     0.4879)        0  
  [    0.4879,     0.5166)        0  
  [    0.5166,     0.5452)      480  ##
  [    0.5452,     0.5739)      240  #
```

### `pred_ee_w_final`  (n=36672)

```
  n=36672  min=0.001194  p01=0.0023017  p10=0.0057188  p25=0.013443  median=0.030913  mean=0.08377  p75=0.12155  p90=0.20623  p99=0.77861  max=1.2028  std=0.13669
```
Histogram (text):

```
  [  0.001194,    0.06128)    25072  ########################################
  [   0.06128,     0.1214)     2428  ####
  [    0.1214,     0.1814)     4262  #######
  [    0.1814,     0.2415)     2649  ####
  [    0.2415,     0.3016)      789  #
  [    0.3016,     0.3617)      193  
  [    0.3617,     0.4218)      177  
  [    0.4218,     0.4819)      162  
  [    0.4819,     0.5419)      143  
  [    0.5419,      0.602)      119  
  [     0.602,     0.6621)      119  
  [    0.6621,     0.7222)      103  
  [    0.7222,     0.7823)       94  
  [    0.7823,     0.8423)       77  
  [    0.8423,     0.9024)       77  
  [    0.9024,     0.9625)       82  
  [    0.9625,      1.023)       39  
  [     1.023,      1.083)       29  
  [     1.083,      1.143)       35  
  [     1.143,      1.203)       23  
```

### `W_ctx_pred_init`  (n=36864)

```
  n=36864  min=6.5843e-07  p01=0.00046483  p10=0.0050814  p25=0.012445  median=0.025121  mean=0.025022  p75=0.037483  p90=0.044945  p99=0.049527  max=0.049999  std=0.014425
```
Histogram (text):

```
  [ 6.584e-07,   0.002501)     1854  ######################################
  [  0.002501,   0.005001)     1763  #####################################
  [  0.005001,     0.0075)     1874  #######################################
  [    0.0075,       0.01)     1840  ######################################
  [      0.01,     0.0125)     1930  ########################################
  [    0.0125,      0.015)     1809  #####################################
  [     0.015,     0.0175)     1835  ######################################
  [    0.0175,       0.02)     1752  ####################################
  [      0.02,     0.0225)     1859  #######################################
  [    0.0225,      0.025)     1821  ######################################
  [     0.025,     0.0275)     1906  ########################################
  [    0.0275,       0.03)     1774  #####################################
  [      0.03,     0.0325)     1875  #######################################
  [    0.0325,      0.035)     1866  #######################################
  [     0.035,     0.0375)     1906  ########################################
  [    0.0375,       0.04)     1882  #######################################
  [      0.04,     0.0425)     1842  ######################################
  [    0.0425,      0.045)     1818  ######################################
  [     0.045,     0.0475)     1825  ######################################
  [    0.0475,       0.05)     1833  ######################################
```

### `W_ctx_pred_final`  (n=36864)

```
  n=36864  min=0.00012303  p01=0.00012358  p10=0.00015635  p25=0.00021948  median=0.0026079  mean=0.015625  p75=0.030541  p90=0.052813  p99=0.066546  max=0.078384  std=0.020597
```
Histogram (text):

```
  [  0.000123,   0.004036)    19200  ########################################
  [  0.004036,   0.007949)     2816  ######
  [  0.007949,    0.01186)      992  ##
  [   0.01186,    0.01578)      784  ##
  [   0.01578,    0.01969)     1040  ##
  [   0.01969,     0.0236)     1264  ###
  [    0.0236,    0.02751)      784  ##
  [   0.02751,    0.03143)      768  ##
  [   0.03143,    0.03534)      768  ##
  [   0.03534,    0.03925)     1264  ###
  [   0.03925,    0.04317)     2064  ####
  [   0.04317,    0.04708)      336  #
  [   0.04708,    0.05099)      432  #
  [   0.05099,    0.05491)     2496  #####
  [   0.05491,    0.05882)      832  ##
  [   0.05882,    0.06273)        0  
  [   0.06273,    0.06664)      736  ##
  [   0.06664,    0.07056)       32  
  [   0.07056,    0.07447)        0  
  [   0.07447,    0.07838)      256  #
```

### `elig_final`  (n=36864)

```
  n=36864  min=0  p01=0  p10=0  p25=3.24e-154  median=1.0814e-150  mean=2.4235  p75=0.041128  p90=8.2194  p99=33.553  max=45.134  std=6.954
```
Histogram (text):

```
  [         0,      2.257)    31232  ########################################
  [     2.257,      4.513)      512  #
  [     4.513,       6.77)      768  #
  [      6.77,      9.027)     1024  #
  [     9.027,      11.28)      256  
  [     11.28,      13.54)      256  
  [     13.54,       15.8)      512  #
  [      15.8,      18.05)        0  
  [     18.05,      20.31)      768  #
  [     20.31,      22.57)      512  #
  [     22.57,      24.82)        0  
  [     24.82,      27.08)        0  
  [     27.08,      29.34)      256  
  [     29.34,      31.59)      256  
  [     31.59,      33.85)      240  
  [     33.85,      36.11)       16  
  [     36.11,      38.36)        0  
  [     38.36,      40.62)        0  
  [     40.62,      42.88)        0  
  [     42.88,      45.13)      256  
```

## ctx_ee structural concentration

```
  fraction of ctx_ee weights >= threshold:
    >=  0.01  90.227%  ####################################
    >=  0.05  49.040%  ####################
    >=  0.10  14.834%  ######
    >=  0.20  8.551%  ###
    >=  0.50  1.963%  #
    >=  1.00  0.000%  
    >=  1.40  0.000%  
```

Interpretation: within-channel init = 1.0, cross-channel init = 0.02. If within-channel weights had stayed near init, ≥25% of the 36672 weights (16 within / 48 per postsyn) would be ≥ 0.5. If only a tiny fraction exceed even 0.1, the within-channel recurrent backbone — the substrate the bump needs — has been dismantled.

## Three-factor gate trajectory (360 gates)

`gate_dw_sum` per-gate aggregate dW deposited on W_ctx_pred (sum over 36864 synapses):

```
  n=360  min=-346.41  p01=-2.3197e-13  p10=-8.7862e-14  p25=-3.699e-14  median=1.3388e-15  mean=-0.96225  p75=3.3399e-14  p90=9.2745e-14  p99=2.0771e-13  max=3.2365e-13  std=18.232
```
Decile trajectory of `gate_dw_sum` across the 360-gate schedule:

```
  | bucket |  trial range |       min |      mean |       max |
  |--------|--------------|-----------|-----------|-----------|
  |    1   |     0..  35  |    -346.4 |    -9.622 | 3.378e-14 |
  |    2   |    36..  71  | -6.127e-14 | -1.263e-15 | 5.839e-14 |
  |    3   |    72.. 107  | -6.14e-14 | 1.655e-15 | 1.185e-13 |
  |    4   |   108.. 143  | -1.525e-13 | 7.472e-16 | 1.867e-13 |
  |    5   |   144.. 179  | -1.887e-13 | -2.605e-15 | 1.584e-13 |
  |    6   |   180.. 215  | -2.358e-13 |  9.16e-16 | 1.638e-13 |
  |    7   |   216.. 251  | -3.509e-13 | 1.394e-15 | 3.236e-13 |
  |    8   |   252.. 287  | -2.293e-13 | -1.556e-15 | 1.749e-13 |
  |    9   |   288.. 323  | -1.233e-13 | -2.468e-15 | 1.799e-13 |
  |   10   |   324.. 359  | -1.951e-13 | 2.154e-15 | 2.202e-13 |
```
Decile trajectory of `gate_elig_mean` (average eligibility at the gate):

```
  | bucket |  trial range |       min |      mean |       max |
  |--------|--------------|-----------|-----------|-----------|
  |    1   |     0..  35  |     26.93 |     46.68 |     74.42 |
  |    2   |    36..  71  |     35.76 |      48.9 |     59.57 |
  |    3   |    72.. 107  |     28.26 |     41.37 |     58.27 |
  |    4   |   108.. 143  |        16 |     35.05 |      49.3 |
  |    5   |   144.. 179  |     16.43 |     32.85 |     49.28 |
  |    6   |   180.. 215  |     18.42 |     31.03 |     47.66 |
  |    7   |   216.. 251  |     9.971 |     25.36 |     39.54 |
  |    8   |   252.. 287  |     8.895 |     19.02 |     37.83 |
  |    9   |   288.. 323  |     5.503 |     13.36 |     23.02 |
  |   10   |   324.. 359  |      5.11 |     9.414 |     15.18 |
```
Decile trajectory of `gate_elig_max` (peak eligibility at the gate):

```
  | bucket |  trial range |       min |      mean |       max |
  |--------|--------------|-----------|-----------|-----------|
  |    1   |     0..  35  |     480.7 |     897.1 |      1261 |
  |    2   |    36..  71  |     471.8 |     805.2 |      1237 |
  |    3   |    72.. 107  |       369 |     612.2 |      1174 |
  |    4   |   108.. 143  |     140.7 |     460.8 |     667.1 |
  |    5   |   144.. 179  |     255.3 |     434.9 |     593.9 |
  |    6   |   180.. 215  |     243.2 |     399.8 |     498.6 |
  |    7   |   216.. 251  |       207 |       330 |     462.5 |
  |    8   |   252.. 287  |     105.2 |     246.8 |     347.7 |
  |    9   |   288.. 323  |     108.9 |     191.1 |     280.9 |
  |   10   |   324.. 359  |     89.26 |     144.4 |     217.5 |
```
Decile trajectory of `gate_w_before` (mean W_ctx_pred before update):

```
  | bucket |  trial range |       min |      mean |       max |
  |--------|--------------|-----------|-----------|-----------|
  |    1   |     0..  35  |   0.01562 |   0.01589 |   0.02502 |
  |    2   |    36..  71  |   0.01562 |   0.01562 |   0.01562 |
  |    3   |    72.. 107  |   0.01562 |   0.01562 |   0.01563 |
  |    4   |   108.. 143  |   0.01562 |   0.01562 |   0.01563 |
  |    5   |   144.. 179  |   0.01562 |   0.01562 |   0.01563 |
  |    6   |   180.. 215  |   0.01562 |   0.01562 |   0.01563 |
  |    7   |   216.. 251  |   0.01562 |   0.01562 |   0.01563 |
  |    8   |   252.. 287  |   0.01562 |   0.01562 |   0.01563 |
  |    9   |   288.. 323  |   0.01562 |   0.01562 |   0.01563 |
  |   10   |   324.. 359  |   0.01562 |   0.01562 |   0.01563 |
```
`gate_n_capped` min/max: 192 / 192 (number of W_ctx_pred synapses at w_max cap per gate — any deviation from 192 would indicate caps forming or releasing).

## Forecast-gate confusion (180 rotating trials)

```
  n_trials_probed       : 180
  ctx_argmax == leader_idx     : 180 / 180  (100.0%)
  pred_argmax == leader_idx    : 178 / 180  (98.9%)
  pred_argmax == expected_trailer : 0 / 180  (0.0%)  [<-- forecast gate]
  ctx_argmax == expected_trailer  : 0 / 180  (0.0%)
  pred_argmax distribution over 6 channels:
    ch0:   34  #################
    ch1:   35  #################
    ch2:   28  ##############
    ch3:   31  ###############
    ch4:   23  ###########
    ch5:   29  ##############
  ctx_argmax distribution over 6 channels:
    ch0:   32  ################
    ch1:   35  #################
    ch2:   28  ##############
    ch3:   31  ###############
    ch4:   25  ############
    ch5:   29  ##############
```

## Preliminary findings (checkpoint only)

1. **W_ctx_pred is collapsed to a fixed point of the row-cap rescale, not "depressed by LTD".** The `gate_dw_sum < 0` readout is an artefact of the rescale: the rule adds a positive `eta × elig × M(t)` LTP increment, `apply_modulatory_update` then rescales every row down to `sum = w_row_max = 3.0`, and the book-keeping `dw_sum = (w_new - w).sum()` after that rescale is negative. The underlying *rule* is not sign-flipped — the cap is eating the increment.

2. **Eligibility traces are healthy at first, decay across schedule.** `gate_elig_mean` in range 5–75, `gate_elig_max` 89–1261, 6× decay late vs early. The three-factor multiplier `eligibility × M(t) × ACh` has non-trivial magnitude; the LTP is being cancelled by the row-cap rescale, not by dead elig.

3. **Forecast probability 0.000 is a deterministic structural zero.** Checkpoint's own `h_argmax_pred` column lets us verify the gate's counting — see confusion block above. Pred tracks the current leader (amplifier, 98.9%), never the expected trailer. Expected when ctx→pred mapping is a uniform-floor flat projection.

4. **The ctx_ee within-channel backbone is also dismantled** (1.96% of weights ≥ 0.5 vs init was 1.0 for within-channel), so even if W_ctx_pred were learning correctly, H_ctx would not have a stable bump to project. This is a *second* failure mode (not just W_ctx_pred) and is consistent with the Vogels iSTDP over-strengthening that the `inh_w_max` knob was aimed at. Both failures must be addressed.

5. **Remaining telemetry** (for completeness but not strictly needed for the root-cause call):
   (a) `ctx.inh→e` and `pred.inh→e` weight histograms at end-of-training to quantify the Vogels over-strengthening;
   (b) per-trial H_ctx / H_pred firing traces across the 360 trials to watch the bump die;
   (c) per-synapse W_ctx_pred trajectory to watch the row-cap collapse in real time.
   These require a short re-run (~15 min at n_trials=60) with SpikeMonitors + weight snapshots. Awaiting Lead's call.
