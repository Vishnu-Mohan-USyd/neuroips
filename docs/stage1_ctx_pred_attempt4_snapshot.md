# Stage-1 ctx_pred attempt #4 — diagnostic snapshot

Checkpoint: `expectation_snn/data/checkpoints/stage_1_ctx_pred_seed42.npz`.
Log: `logs/train_stage1_ctx_pred_full_seed42_attempt4.log`.

Attempt #4 (driver commit `d8ca59f`, Fix A `0f30cd2`, Fix B(i)
`b5d8fd7`, Fix C `b5ba400`) was Lead's compound-fix pass following
the task-#47 Debugger H1+H3+H4 verdict. The three intended
simulator changes were:

- Fix A: `DEFAULT_TAU_COINC_MS` 20 → 500 ms (span leader→trailer gap).
- Fix B(i): `HRingConfig.inh_rho_hz` 2 → 20 Hz (Vogels target rate).
- Fix C: `DEFAULT_W_TARGET` 0.05 → 0.0075 (match post-init mean).

## Gate verdict — FAIL

| check | value | band | result |
|---|---|---|---|
| h_bump_persistence_ms | 10.0 (probe 0.0) | [200, 500] | **FAIL** |
| h_preprobe_forecast_prob | 0.039 | [0.25, 1.00] | **FAIL** |
| no_runaway (ctx_e_rate) | 21.27 Hz | [0, 80] | PASS |

## Critical finding: Fix B(i) was silently reverted

The attempt-#4 evidence-log JSON reports
`"HRingConfig.inh_rho_hz": 10.0` for the same `H_CFG` object whose
startup banner (earlier in the same run) printed `20.0 Hz`. Root
cause: `expectation_snn/brian2_model/train.py` line 553 has
`cfg.inh_rho_hz = 10.0` inside `_stage1_h_cfg`, and the helper
mutates the passed `cfg` in-place (line 536: `cfg = base or
HRingConfig()`), so the driver's `H_CFG` got its field overwritten
before Brian2 Synapses were built.

This is the **same failure pattern** as attempt #2, where
`HRingConfig(inh_w_max=2.0)` was clobbered to `1.5` by
`_stage1_h_cfg` line 556. Two silent overrides in four attempts —
it is a pattern, not a one-off. The Stage-1 helper is treated as
canonical by every Stage-1 driver; config changes intended for
Stage-1 must land inside the helper, not at the dataclass default.

Implication for the task-#47 Debugger verdict: the Debugger
assumed attempts #1-#3 ran with `inh_rho_hz = 2.0` (the dataclass
default). They actually ran with `10.0` (the helper override). So
the observed target/actual mismatch was 10 Hz / 21 Hz ≈ 2×, not
2 Hz / 21 Hz ≈ 10×. H3 (Vogels iSTDP runaway) is still the right
family of hypothesis — the H3 sanity run (bg `b1od0brws`, commit-
local monkeypatch `inh_eta = 0`) saw persistence lift from ~10 ms
to 990 ms with the 2.0 dataclass default active — but the
*quantitative* expectation for Fix B(i) was built on a stale
baseline.

Fix A and Fix C landed correctly: the runtime log line
`ctx_pred: tau_coinc=500ms tau_elig=1000ms eta=1.00e-03
w_target=0.007` confirms both reached the Brian2 build. These
fields live on `HContextPredictionConfig`, which `_stage1_h_cfg`
does not touch.

## Comparison across attempts

| metric | #1/#2 | #3 | #4 |
|---|---|---|---|
| `W_ctx_pred_init` mean | 0.02502 | 0.00751 | 0.00751 |
| `W_ctx_pred_final` mean | 0.01562 (=3/192) | 0.01562 (=3/192) | 0.01562 (=3/192) |
| `W_ctx_pred_final` max | ~0.05 | 0.0707 | 0.0758 |
| `elig_final` mean | ~1 | 3.03 | 34.12 |
| `ctx_e_rate_hz` | 21.27 | 21.27 | 21.27 |
| `pred_e_rate_hz` | ~19 | 19.23 | 17.89 |
| `h_bump_persistence_ms` | 10 | 10 | 10 |
| `h_preprobe_forecast_prob` | 0.000 | 0.139 | **0.039** |

Readouts across the table:

1. The row-cap floor `3.0/192 = 0.015625` is the dominant attractor
   for `W_ctx_pred_final` mean across all four attempts — Fix A
   (tau_coinc 500 ms) and Fix C (w_target 0.0075) did not release
   the mean from it, although they widened the upper tail
   (max 0.05 → 0.076).
2. `elig_final` mean grew ~11× from #3 → #4, consistent with the
   25× longer τ_coinc (Fix A) producing more overlap between
   `x_pre` residuals and post-spikes. Eligibility is not the
   bottleneck.
3. `ctx_e_rate_hz = 21.27` is **bit-identical across all four
   attempts** — direct confirmation that the ring's E/I state was
   not perturbed by any fix. Fix B(i) would have shifted this
   rate (target 20 Hz vs helper-overridden 10 Hz); it didn't.
4. Forecast **dropped** from 0.139 (#3) → 0.039 (#4). Fix A + Fix
   C together made the forecast worse. Hypothesis (untested):
   with τ_coinc spanning the full leader epoch, the rule
   accumulates more *leader-leader* coincidence eligibility
   because the V1 → H_pred teacher lights up the leader channel
   in pred during the leader window. The gate then consolidates
   the leader → leader mapping, not leader → expected-trailer.
   The V1 → H_pred always-on teacher is the candidate
   architectural culprit.

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
  n=36672  min=0.0011182  p01=0.0015019  p10=0.0055706  p25=0.010452  median=0.021832  mean=0.08377  p75=0.10744  p90=0.21687  p99=0.83721  max=1.5009  std=0.15479
```
Histogram (text):

```
  [  0.001118,    0.07611)    26140  ########################################
  [   0.07611,     0.1511)     3726  ######
  [    0.1511,     0.2261)     3541  #####
  [    0.2261,     0.3011)     1570  ##
  [    0.3011,     0.3761)      334  #
  [    0.3761,     0.4511)      244  
  [    0.4511,     0.5261)      175  
  [    0.5261,      0.601)      155  
  [     0.601,      0.676)      165  
  [     0.676,      0.751)      139  
  [     0.751,      0.826)       99  
  [     0.826,      0.901)       76  
  [     0.901,      0.976)       66  
  [     0.976,      1.051)       53  
  [     1.051,      1.126)       42  
  [     1.126,      1.201)       28  
  [     1.201,      1.276)       37  
  [     1.276,      1.351)       43  
  [     1.351,      1.426)       13  
  [     1.426,      1.501)       26  
```

### `W_ctx_pred_init`  (n=36864)

```
  n=36864  min=1.9753e-07  p01=0.00013945  p10=0.0015244  p25=0.0037336  median=0.0075362  mean=0.0075066  p75=0.011245  p90=0.013484  p99=0.014858  max=0.015  std=0.0043275
```
Histogram (text):

```
  [ 1.975e-07,  0.0007502)     1854  ######################################
  [ 0.0007502,     0.0015)     1763  #####################################
  [    0.0015,    0.00225)     1874  #######################################
  [   0.00225,      0.003)     1840  ######################################
  [     0.003,    0.00375)     1930  ########################################
  [   0.00375,     0.0045)     1809  #####################################
  [    0.0045,    0.00525)     1835  ######################################
  [   0.00525,      0.006)     1752  ####################################
  [     0.006,    0.00675)     1859  #######################################
  [   0.00675,     0.0075)     1821  ######################################
  [    0.0075,    0.00825)     1906  ########################################
  [   0.00825,      0.009)     1774  #####################################
  [     0.009,    0.00975)     1875  #######################################
  [   0.00975,     0.0105)     1866  #######################################
  [    0.0105,    0.01125)     1906  ########################################
  [   0.01125,      0.012)     1882  #######################################
  [     0.012,    0.01275)     1842  ######################################
  [   0.01275,     0.0135)     1818  ######################################
  [    0.0135,    0.01425)     1825  ######################################
  [   0.01425,      0.015)     1833  ######################################
```

### `W_ctx_pred_final`  (n=36864)

```
  n=36864  min=0.00017499  p01=0.00021884  p10=0.00059342  p25=0.0013294  median=0.0054872  mean=0.015624  p75=0.027249  p90=0.047345  p99=0.069981  max=0.075841  std=0.018637
```
Histogram (text):

```
  [  0.000175,   0.003958)    16640  ########################################
  [  0.003958,   0.007742)     3328  ########
  [  0.007742,    0.01152)     1536  ####
  [   0.01152,    0.01531)     2304  ######
  [   0.01531,    0.01909)     2048  #####
  [   0.01909,    0.02287)      736  ##
  [   0.02287,    0.02666)      784  ##
  [   0.02666,    0.03044)     1552  ####
  [   0.03044,    0.03422)     1008  ##
  [   0.03422,    0.03801)     1184  ###
  [   0.03801,    0.04179)     1392  ###
  [   0.04179,    0.04557)      240  #
  [   0.04557,    0.04936)     1264  ###
  [   0.04936,    0.05314)      688  ##
  [   0.05314,    0.05692)      848  ##
  [   0.05692,    0.06071)       32  
  [   0.06071,    0.06449)      720  ##
  [   0.06449,    0.06827)       48  
  [   0.06827,    0.07206)      256  #
  [   0.07206,    0.07584)      256  #
```

### `elig_final`  (n=36864)

```
  n=36864  min=0  p01=0  p10=0  p25=4.2963e-05  median=0.00016781  mean=34.122  p75=38.7  p90=104.92  p99=290.03  max=367.34  std=64.837
```
Histogram (text):

```
  [         0,      18.37)    22784  ########################################
  [     18.37,      36.73)     4400  ########
  [     36.73,       55.1)     2224  ####
  [      55.1,      73.47)     2048  ####
  [     73.47,      91.84)      992  ##
  [     91.84,      110.2)      944  ##
  [     110.2,      128.6)      112  
  [     128.6,      146.9)      544  #
  [     146.9,      165.3)      752  #
  [     165.3,      183.7)       16  
  [     183.7,        202)      512  #
  [       202,      220.4)        0  
  [     220.4,      238.8)      720  #
  [     238.8,      257.1)       48  
  [     257.1,      275.5)        0  
  [     275.5,      293.9)      496  #
  [     293.9,      312.2)       16  
  [     312.2,      330.6)        0  
  [     330.6,        349)        0  
  [       349,      367.3)      256  
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

## Three-factor gate trajectory (360 gates)

`gate_dw_sum` per-gate aggregate dW:

```
  n=360  min=-0.03811  p01=-0.031175  p10=-0.018681  p25=-0.0099755  median=-2.8705e-13  mean=0.83127  p75=0.0061049  p90=0.020707  p99=20.945  max=150.06  std=9.0501
```
Decile trajectory of `gate_dw_sum`:

```
  | bucket |  trial range |       min |      mean |       max |
  |--------|--------------|-----------|-----------|-----------|
  |    1   |     0..  35  |  -0.03118 |     8.312 |     150.1 |
  |    2   |    36..  71  |  -0.03102 | 0.0004911 |   0.08219 |
  |    3   |    72.. 107  |  -0.01871 | 0.0007526 |   0.06746 |
  |    4   |   108.. 143  |  -0.01871 | -0.0003013 |   0.02485 |
  |    5   |   144.. 179  |   -0.0249 | 0.0001288 |   0.05562 |
  |    6   |   180.. 215  |  -0.03579 | 0.0001726 |   0.06642 |
  |    7   |   216.. 251  |  -0.03811 |  3.42e-15 |    0.0748 |
  |    8   |   252.. 287  |  -0.01871 | -4.42e-15 |   0.05589 |
  |    9   |   288.. 323  |  -0.02978 | 1.222e-15 |   0.05861 |
  |   10   |   324.. 359  |  -0.03277 | -0.0005665 |   0.03757 |
```
Decile trajectory of `gate_elig_mean`:

```
  | bucket |  trial range |       min |      mean |       max |
  |--------|--------------|-----------|-----------|-----------|
  |    1   |     0..  35  |       234 |      1078 |      1670 |
  |    2   |    36..  71  |     502.7 |      1031 |      1481 |
  |    3   |    72.. 107  |     334.2 |     803.8 |      1278 |
  |    4   |   108.. 143  |     156.5 |     564.1 |     927.9 |
  |    5   |   144.. 179  |       151 |     487.3 |     808.6 |
  |    6   |   180.. 215  |     143.9 |     374.7 |     619.8 |
  |    7   |   216.. 251  |     141.7 |     339.1 |     670.6 |
  |    8   |   252.. 287  |     131.9 |     317.9 |     722.5 |
  |    9   |   288.. 323  |     53.61 |     293.1 |     590.4 |
  |   10   |   324.. 359  |     36.42 |     191.6 |     445.3 |
```
Decile trajectory of `gate_elig_max`:

```
  | bucket |  trial range |       min |      mean |       max |
  |--------|--------------|-----------|-----------|-----------|
  |    1   |     0..  35  |      4659 | 2.165e+04 | 2.927e+04 |
  |    2   |    36..  71  |      6635 | 1.795e+04 | 2.676e+04 |
  |    3   |    72.. 107  |      5314 | 1.166e+04 | 2.653e+04 |
  |    4   |   108.. 143  |      2521 |      6478 | 1.104e+04 |
  |    5   |   144.. 179  |      2952 |      5671 |      9479 |
  |    6   |   180.. 215  |      2166 |      4539 |      8100 |
  |    7   |   216.. 251  |      2172 |      3989 |      6560 |
  |    8   |   252.. 287  |      2054 |      3883 |      9157 |
  |    9   |   288.. 323  |      1032 |      3239 |      7389 |
  |   10   |   324.. 359  |     744.3 |      2417 |      4967 |
```
Decile trajectory of `gate_w_before`:

```
  | bucket |  trial range |       min |      mean |       max |
  |--------|--------------|-----------|-----------|-----------|
  |    1   |     0..  35  |  0.007507 |   0.01503 |   0.01562 |
  |    2   |    36..  71  |   0.01562 |   0.01562 |   0.01563 |
  |    3   |    72.. 107  |   0.01562 |   0.01562 |   0.01563 |
  |    4   |   108.. 143  |   0.01562 |   0.01562 |   0.01563 |
  |    5   |   144.. 179  |   0.01562 |   0.01562 |   0.01563 |
  |    6   |   180.. 215  |   0.01562 |   0.01562 |   0.01563 |
  |    7   |   216.. 251  |   0.01562 |   0.01562 |   0.01563 |
  |    8   |   252.. 287  |   0.01562 |   0.01562 |   0.01563 |
  |    9   |   288.. 323  |   0.01562 |   0.01562 |   0.01563 |
  |   10   |   324.. 359  |   0.01562 |   0.01562 |   0.01563 |
```
`gate_n_capped` min/max: 48 / 192 (presyn rows whose outgoing row-sum triggered the `w_row_max = 3.0` hard rescale that gate).

## Forecast-gate confusion (180 rotating trials)

```
  n_trials_probed       : 180
  ctx_argmax == leader_idx        : 180 / 180  (100.0%)
  pred_argmax == leader_idx       : 166 / 180  (92.2%)  [amplifier signature]
  pred_argmax == expected_trailer : 7 / 180  (3.9%)  [<-- forecast gate]
  ctx_argmax == expected_trailer  : 0 / 180  (0.0%)
  pred_argmax distribution over 6 channels:
    ch0:   39  ###################
    ch1:   33  ################
    ch2:   27  #############
    ch3:   30  ###############
    ch4:   28  ##############
    ch5:   23  ###########
  ctx_argmax distribution over 6 channels:
    ch0:   32  ################
    ch1:   35  #################
    ch2:   28  ##############
    ch3:   31  ###############
    ch4:   25  ############
    ch5:   29  ##############
```

## Iteration status

Per Lead's attempt-#4 dispatch: **0 iterations remaining** after
this run. No attempt #5 will be launched without user approval.
Lead is escalating the full evidence snapshot to the user; this
document is the artefact of record for that escalation.
