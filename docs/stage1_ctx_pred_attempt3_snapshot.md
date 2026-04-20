# Stage-1 ctx_pred attempt #3 — diagnostic snapshot

Checkpoint: `expectation_snn/data/checkpoints/stage_1_ctx_pred_seed42.npz`
(overwritten by attempt #3; the attempt #1/#2 analysis is preserved at
`docs/stage1_ctx_pred_attempt1_snapshot.md` and was computed from the
older state captured before this run).

Attempt #3 applied the `w_init_frac: 0.05 -> 0.015` fix (commit `5ef422b`)
that commit `4493fa8`'s snapshot identified as the root cause of the
init-time row-cap collapse. The fix landed correctly — init mean is now
`0.007506` (verified below; attempts #1/#2 were `0.025022`) — but Stage-1
still FAILs with the identical verdict triple.

Gate verdict (from `logs/train_stage1_ctx_pred_full_seed42_attempt3.log`):
`h_bump_persistence_ms = 10.0` FAIL, `h_preprobe_forecast_prob = 0.0` FAIL,
`no_runaway = 21.274` PASS. The same FAIL pattern as attempts #1/#2, but
the *mechanism* leading there is visibly different — see the trajectory
section below.

The effective HRingConfig is the same as attempts #1/#2 (`inh_w_max = 1.5`,
forced by `_stage1_h_cfg`), so any change between this snapshot and the
attempt-#1 snapshot is attributable solely to `w_init_frac`.

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
  n=36672  min=0.0006573  p01=0.0011085  p10=0.0057745  p25=0.013277  median=0.034564  mean=0.08377  p75=0.10358  p90=0.19166  p99=0.77794  max=1.3967  std=0.13672
```
Histogram (text):

```
  [ 0.0006573,    0.07046)    25274  ########################################
  [   0.07046,     0.1403)     3776  ######
  [    0.1403,     0.2101)     4714  #######
  [    0.2101,     0.2799)     1119  ##
  [    0.2799,     0.3497)      494  #
  [    0.3497,     0.4195)      210  
  [    0.4195,     0.4893)      170  
  [    0.4893,     0.5591)      160  
  [    0.5591,     0.6289)      139  
  [    0.6289,     0.6987)      134  
  [    0.6987,     0.7685)       99  
  [    0.7685,     0.8383)      104  
  [    0.8383,     0.9081)       80  
  [    0.9081,     0.9779)       54  
  [    0.9779,      1.048)       49  
  [     1.048,      1.118)       43  
  [     1.118,      1.187)       26  
  [     1.187,      1.257)       12  
  [     1.257,      1.327)       12  
  [     1.327,      1.397)        3  
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
  n=36864  min=0.00010552  p01=0.00010574  p10=0.00016356  p25=0.00021637  median=0.0021905  mean=0.015625  p75=0.032281  p90=0.051278  p99=0.063922  max=0.070727  std=0.020459
```
Histogram (text):

```
  [ 0.0001055,   0.003637)    19456  ########################################
  [  0.003637,   0.007168)     1792  ####
  [  0.007168,     0.0107)     1280  ###
  [    0.0107,    0.01423)     1024  ##
  [   0.01423,    0.01776)     1008  ##
  [   0.01776,    0.02129)      784  ##
  [   0.02129,    0.02482)      960  ##
  [   0.02482,    0.02835)      576  #
  [   0.02835,    0.03189)      496  #
  [   0.03189,    0.03542)     1520  ###
  [   0.03542,    0.03895)     1296  ###
  [   0.03895,    0.04248)      688  #
  [   0.04248,    0.04601)      800  ##
  [   0.04601,    0.04954)      832  ##
  [   0.04954,    0.05307)     1024  ##
  [   0.05307,     0.0566)     1168  ##
  [    0.0566,    0.06013)     1344  ###
  [   0.06013,    0.06366)      304  #
  [   0.06366,     0.0672)      320  #
  [    0.0672,    0.07073)      192  
```

### `elig_final`  (n=36864)

```
  n=36864  min=0  p01=0  p10=1.3622e-162  p25=5.2867e-154  median=2.4637e-11  mean=3.0314  p75=0.48579  p90=7.5038  p99=40.766  max=48.863  std=8.4322
```
Histogram (text):

```
  [         0,      2.443)    29696  ########################################
  [     2.443,      4.886)     2304  ###
  [     4.886,      7.329)     1024  #
  [     7.329,      9.773)      512  #
  [     9.773,      12.22)      512  #
  [     12.22,      14.66)      256  
  [     14.66,       17.1)      256  
  [      17.1,      19.55)        0  
  [     19.55,      21.99)        0  
  [     21.99,      24.43)      432  #
  [     24.43,      26.87)      336  
  [     26.87,      29.32)        0  
  [     29.32,      31.76)        0  
  [     31.76,       34.2)      768  #
  [      34.2,      36.65)        0  
  [     36.65,      39.09)      256  
  [     39.09,      41.53)      256  
  [     41.53,      43.98)        0  
  [     43.98,      46.42)        0  
  [     46.42,      48.86)      256  
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
  n=360  min=-2.9204e-13  p01=-2.1843e-13  p10=-8.6991e-14  p25=-3.907e-14  median=2.6203e-15  mean=0.83133  p75=4.6659e-14  p90=1.0443e-13  p99=24.362  max=146.48  std=8.9262
```
Decile trajectory of `gate_dw_sum` across the 360-gate schedule:

```
  | bucket |  trial range |       min |      mean |       max |
  |--------|--------------|-----------|-----------|-----------|
  |    1   |     0..  35  | -4.611e-14 |     8.313 |     146.5 |
  |    2   |    36..  71  | -7.52e-14 | 6.714e-16 | 5.717e-14 |
  |    3   |    72.. 107  | -6.842e-14 | -2.288e-15 | 6.134e-14 |
  |    4   |   108.. 143  | -1.186e-13 | 2.153e-15 | 1.722e-13 |
  |    5   |   144.. 179  | -1.672e-13 | 1.835e-16 |  2.13e-13 |
  |    6   |   180.. 215  | -1.25e-13 | 1.723e-16 | 1.392e-13 |
  |    7   |   216.. 251  | -2.594e-13 | 9.564e-16 | 2.349e-13 |
  |    8   |   252.. 287  | -2.92e-13 | -4.978e-16 | 2.592e-13 |
  |    9   |   288.. 323  | -2.007e-13 | -1.686e-15 | 1.971e-13 |
  |   10   |   324.. 359  | -1.761e-13 | 7.621e-16 | 2.098e-13 |
```
Decile trajectory of `gate_elig_mean` (average eligibility at the gate):

```
  | bucket |  trial range |       min |      mean |       max |
  |--------|--------------|-----------|-----------|-----------|
  |    1   |     0..  35  |     27.91 |     47.82 |     71.66 |
  |    2   |    36..  71  |     22.12 |     46.56 |     58.56 |
  |    3   |    72.. 107  |     25.91 |     40.63 |      58.6 |
  |    4   |   108.. 143  |     26.58 |     35.91 |     48.03 |
  |    5   |   144.. 179  |     18.96 |     33.92 |     46.99 |
  |    6   |   180.. 215  |     17.66 |     31.75 |     48.21 |
  |    7   |   216.. 251  |     12.55 |     26.51 |     36.63 |
  |    8   |   252.. 287  |     9.477 |     20.19 |     37.17 |
  |    9   |   288.. 323  |     5.588 |     14.54 |     23.84 |
  |   10   |   324.. 359  |     4.478 |     10.42 |     17.14 |
```
Decile trajectory of `gate_elig_max` (peak eligibility at the gate):

```
  | bucket |  trial range |       min |      mean |       max |
  |--------|--------------|-----------|-----------|-----------|
  |    1   |     0..  35  |     515.9 |     901.9 |      1287 |
  |    2   |    36..  71  |       429 |       805 |      1210 |
  |    3   |    72.. 107  |     344.7 |     602.8 |      1152 |
  |    4   |   108.. 143  |       280 |     460.1 |     656.5 |
  |    5   |   144.. 179  |     319.6 |     436.9 |     523.6 |
  |    6   |   180.. 215  |     235.5 |     403.8 |     534.7 |
  |    7   |   216.. 251  |       236 |     331.7 |     434.7 |
  |    8   |   252.. 287  |       125 |     260.1 |     382.7 |
  |    9   |   288.. 323  |     112.4 |       207 |     333.6 |
  |   10   |   324.. 359  |     96.99 |     148.9 |     225.8 |
```
Decile trajectory of `gate_w_before` (mean W_ctx_pred before update):

```
  | bucket |  trial range |       min |      mean |       max |
  |--------|--------------|-----------|-----------|-----------|
  |    1   |     0..  35  |  0.007507 |   0.01502 |   0.01562 |
  |    2   |    36..  71  |   0.01562 |   0.01562 |   0.01562 |
  |    3   |    72.. 107  |   0.01562 |   0.01562 |   0.01562 |
  |    4   |   108.. 143  |   0.01562 |   0.01562 |   0.01563 |
  |    5   |   144.. 179  |   0.01562 |   0.01562 |   0.01563 |
  |    6   |   180.. 215  |   0.01562 |   0.01562 |   0.01563 |
  |    7   |   216.. 251  |   0.01562 |   0.01562 |   0.01563 |
  |    8   |   252.. 287  |   0.01562 |   0.01562 |   0.01563 |
  |    9   |   288.. 323  |   0.01562 |   0.01562 |   0.01563 |
  |   10   |   324.. 359  |   0.01562 |   0.01562 |   0.01563 |
```
`gate_n_capped` min/max: 48 / 192 (number of W_ctx_pred synapses at w_max cap per gate — any deviation from 192 would indicate caps forming or releasing).

## Forecast-gate confusion (180 rotating trials)

```
  n_trials_probed       : 180
  ctx_argmax == leader_idx     : 180 / 180  (100.0%)
  pred_argmax == leader_idx    : 179 / 180  (99.4%)
  pred_argmax == expected_trailer : 0 / 180  (0.0%)  [<-- forecast gate]
  ctx_argmax == expected_trailer  : 0 / 180  (0.0%)
  pred_argmax distribution over 6 channels:
    ch0:   33  ################
    ch1:   35  #################
    ch2:   28  ##############
    ch3:   31  ###############
    ch4:   24  ############
    ch5:   29  ##############
  ctx_argmax distribution over 6 channels:
    ch0:   32  ################
    ch1:   35  #################
    ch2:   28  ##############
    ch3:   31  ###############
    ch4:   25  ############
    ch5:   29  ##############
```

## Findings for attempt #3 (post-fix, checkpoint only)

1. **The `w_init_frac` fix landed as intended.** `W_ctx_pred_init`
   summary: `mean = 0.007506`, `max = 0.0150` (uniform(0, 0.015)); init
   row sum = `0.0075 × 192 = 1.44`, well below `w_row_max = 3.0`.
   Attempts #1/#2 had init mean `0.0250` / row sum `4.80`.

2. **Bucket-1 learning was real.** `gate_dw_sum` bucket 1 (trials 0-35)
   has `min = -4.6e-14`, `mean = 8.3`, **`max = 146.5`** — six orders of
   magnitude above the numerical-noise floor seen in buckets 2-10
   (`|dw_sum|` ≤ ~3e-13). Attempt-#1/#2 bucket 1 had `max ≈ 3e-14`
   (numerical noise only) and `min = -346` (the immediate-collapse
   rescale loss). **Attempt #3 has ~35 trials of genuine LTP before the
   cap takes over.**

3. **But the collapse still happens, just delayed.** `W_ctx_pred_final`
   mean is **exactly** `0.015625 = 3.0/192` — the row-cap fixed point.
   `gate_w_before` trajectory shows the collapse completing by trial ~36:
   bucket 1 `min=0.007507 → max=0.01562` (ramping up), bucket 2+ pinned at
   `0.01562`. `gate_n_capped` varies 48-192 across gates (vs constant 192
   in attempts #1/#2) — the cap is less universal but still firing on
   every gate.

4. **The `w_target = 0.05` decay is a uniform UP pump.** With current
   per-synapse mean at ~0.0075, the decay term
   `-γ × (w - w_target) × dt_trial` is positive and roughly uniform
   across all 36864 synapses, driving them toward `w_target = 0.05`
   (6.7× the actual per-synapse mean, 3.3× the init max 0.015). This
   pushes row sums into the cap independently of whether eligibility is
   non-zero — even "learning-dead" rows get pumped up.

5. **Amplifier-not-predictor signature persists.** Forecast confusion:
   `pred_argmax == leader: 179/180 (99.4%)`, `== expected_trailer: 0/180
   (0.0%)`. The ~35 trials of bucket-1 LTP consolidated a *flat
   ctx→pred projection that amplifies whichever ctx channel is
   currently bumping* — which during the pre-trailer probe is the
   leader, never the trailer. This persists across the three attempts
   and is not sensitive to the cap regime (same signature whether LTP
   was suppressed from trial 1 or ran for 35 trials).

6. **Bump persistence still 10ms.** The H_ctx ring dynamics
   (`inh_w_max = 1.5`, Vogels iSTDP) produce the same 10-ms bump
   collapse as attempts #1/#2; `w_init_frac` does not touch this.
   This is a separate (pre-existing) H_ctx failure that must be
   addressed alongside any `W_ctx_pred` rule fix.

**Debugger hypothesis space** (compound):
- H1: `τ_coinc = 20 ms` ≪ 500-ms leader→trailer gap — rule cannot
  bridge the temporal offset even with eligibility.
- H2: Hard-rescale row-cap destroys bucket-1 heterogeneity; researcher
  spec was soft-clip.
- H3: H_ctx bump-persistence bottleneck is orthogonal — iSTDP over-
  strengthening inh→E at `inh_w_max = 1.5` after 360 trials.
- H4: `w_target = 0.05` uniform UP pump compounds the cap.
