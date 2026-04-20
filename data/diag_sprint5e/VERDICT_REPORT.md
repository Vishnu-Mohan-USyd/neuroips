# Sprint 5e-Diag — Verdict report

Task #43. Evidence-only verification of three reviewer-claimed upstream bugs
and a critical H-only forecast unit test. Pre-reg in
`expectation_snn/docs/SPRINT_5D_POST_VERDICT_REVIEW.md`. All deliverables
saved under `data/diag_sprint5e/`; scripts under `scripts/`.

## Bug 1 — Richter training schedule has zero predictive contingency — **CONFIRMED**

**Code evidence.** `expectation_snn/brian2_model/stimulus.py` lines 199–207:
```python
pairs = np.empty((n_trials, 2), dtype=np.int64)
base = np.array([(i, j) for i in range(n_orients) for j in range(n_orients)],
                dtype=np.int64)                                 # (36, 2)
for r in range(replicates):
    block = base.copy()
    rng.shuffle(block, axis=0)
    pairs[r * n_pairs : (r + 1) * n_pairs] = block
```
Each 36-trial block is a permutation of **all 36 leader–trailer pairs**; L.200
builds an exhaustive Cartesian product. Docstring (L160-161) confirms: "all
36 pairs appear equally often (balanced across repeats)". Asserted at
L194-198.

**Empirical measurement** (`scripts/diag_richter_schedule_stats.py`, seed=42,
`n_trials=360`):
- Current schedule: `max_L max_T P(T|L) = 0.1667` (exactly 1/6), entropy
  `log₂(6) = 2.585` bits (max possible). **Zero statistical regularity.**
- Reference biased deranged-permutation `f(L)=(L+1)%6`, `P(L→f(L))=0.80`:
  `max P(T|L) = 0.883`, min-L entropy = 0.735 bits.

**Validator** (`scripts/validate_richter_training_statistics.py`, threshold
`max P(T|L) ≥ 0.70` and min-entropy ≤ log₂(6) − 0.5):
- current schedule: **FAIL** (max=0.167, entropy=2.585)
- biased reference: **PASS** (max=0.883, entropy=0.735)

STDP on H_R cannot learn a forecast from a uniform transition matrix — the
training objective literally offers no signal to minimise against.

## Bug 2 — Stage-1 MI gate measures post-trailer, not pre-trailer, H — **CONFIRMED**

**Code evidence.** `expectation_snn/validation/stage_1_gate.py` L7, L161:
> `check_h_transition_mi` : MI(leader_idx, H_R_argmax_at_+500ms) > 0

Implementation in `expectation_snn/brian2_model/train.py`:
- L628-629: defaults `probe_delay_ms=500.0`, `probe_window_ms=100.0`.
- L720-722:
  ```python
  trailer_end_abs = trial_start_abs + leader_ms + trailer_ms
  probe_win_start = trailer_end_abs + probe_delay_ms - probe_window_ms / 2.0
  probe_win_end   = trailer_end_abs + probe_delay_ms + probe_window_ms / 2.0
  ```
- L729-740: per-trial H argmax computed in `[trailer_end+450, trailer_end+550]` ms
  and passed to `check_h_transition_mi` (L817-819).

The probe window is **after** the trailer was presented. An H that simply
holds a bump at the trailer channel (i.e. responds to the current input)
passes MI against leader_idx iff leader→trailer has contingency in the
training set — and under Bug 1 it does not. More critically: a pre-trailer
forecast gate is never evaluated anywhere in the training pipeline.

**Validator** (`scripts/validate_stage1_preprobe_forecast.py`, reuses Sprint
5d D1 pre-probe rates; last 100 ms of the leader; threshold
`P(argmax == expected_trailer_ch) ≥ 0.25`, chance=1/6=0.167):

| seed | P(amax=expected_trailer) | P(amax=leader) | verdict |
|------|--------------------------|----------------|---------|
| 42   | 0.015                    | 0.765          | **FAIL** |
| 43   | 0.015                    | 0.795          | **FAIL** |
| 44   | 0.030                    | 0.765          | **FAIL** |

The current Stage-1 checkpoint carries **no pre-trailer forecast** — argmax
is locked on the leader (memory of the still-active cue) 77–80% of the time,
and hits the expected-trailer channel well **below chance**. This is exactly
the post-trailer-gate pathology the reviewer predicted: training never
required H to produce a pre-trailer prediction, so it doesn't.

## Bug 3 — Tang forecast is underdetermined: no direction state — **CONFIRMED**

**Code evidence.** `expectation_snn/brian2_model/h_ring.py`:
- L60: `N_CHANNELS = 12` (orientation only).
- `grep -iE 'direction|dir_|cw|ccw|rotation_dir' h_ring.py` returns **0
  matches**. The H ring's state variables are orientation-channel spike
  counts / recurrent conductances only.

**Tang task-side degeneracy** — `expectation_snn/brian2_model/stimulus.py`
L314-316 (`tang_rotating_sequence`):
```python
start = int(rng.integers(0, n_orients))
direction = int(rng.choice((-1, +1)))     # per-block
block_len = int(rng.integers(blen_lo, blen_hi + 1))
for k in range(block_len):
    idx = (start + direction * k) % n_orients
```

Direction is chosen per block. Given a current orientation `o_curr`, the
next orientation is `(o_curr + 1) % 6` under CW (+1) but `(o_curr − 1) % 6`
under CCW (−1). Concrete example trials from the same `tang_rotating_sequence`:

| current orient idx | block direction | next orient idx (expected) |
|---|---|---|
| 2 | +1 (CW)  | 3 |
| 2 | −1 (CCW) | 1 |
| 5 | +1 (CW)  | 0 |
| 5 | −1 (CCW) | 4 |

Same current state, different expected next state — the mapping `o_curr →
o_next` is not a function of orientation alone; it is a function of
`(orientation, direction)`. H_T, lacking any direction variable, cannot
express the distinction. Tang forecast is formally underdetermined.

## B4 — corrected validators

Both shipped under `scripts/` (no retraining required):
1. `validate_richter_training_statistics.py` — passes on biased schedule,
   fails on current. Confirmed above.
2. `validate_stage1_preprobe_forecast.py` — fails on current Sprint 5d D1
   seeds 42/43/44. Confirmed above.

## B5 — H-only forecast unit test — **critical hinge**

`scripts/diag_h_only_forecast.py`: H_R ring alone (no V1, no feedback routes,
no assays), built via `build_h_r` with production Stage-1 config
(`_stage1_h_cfg`), trained on a biased deranged-permutation schedule
(`P(L→(L+1)%6)=0.80`, 360 trials, seed=42, 10 s pre-settle, ITI=1500 ms),
E↔E STDP plastic with the standard post-syn normalizer. Schedule bias
verified in output: `P(actual_trailer = f(L)) = 0.800`.

Wall time: 1475 s. Ran to completion, no errors.

Measurement: argmax of per-channel H_R rate over the last 180 trials in two
windows — the reviewer's pre-trailer forecast window (last 100 ms of
leader) and the legacy post-trailer window (trailer_end + 500 ms ± 50 ms).

| window | P(amax=leader) | P(amax=f(L)) | P(amax=actual_T) |
|---|---|---|---|
| pre-trailer (forecast)  | **1.000** | **0.000** | 0.050 |
| post-trailer (+500 ms)  | 0.178     | 0.178     | 0.150 |

Pre-trailer argmax is **locked on the leader channel every single trial**
(entropy of the argmax distribution = 2.571 bits ≈ log₂(6) = 2.585 because
leader is uniform over 6 channels). The biased schedule does not induce
any pre-trailer forecast — the ring simply tracks the currently-active cue.
Post-trailer argmax is effectively noise: entropy 0.050 bits, argmax
degenerate on channels 0/1 — the bump has decayed to silence by +500 ms
under the stronger Vogels iSTDP settling of an H-only run, so no window
carries a signal either.

### Interpretation

The reviewer's hinge question — "does biased schedule + correct window
produce forecast under the current architecture?" — is answered **no**.
Even with:
- 80 % contingency on `L → f(L)` baked into the training signal,
- pair STDP on AMPA recurrence (and NMDA co-release) unchanged from
  production cfg,
- a 10 s pre-settle, 360 training trials, post-syn normalisation on E→E,

the pre-probe argmax never leaves the leader channel. The single-ring
Wang attractor is architecturally unable to form a pre-trailer forecast:
the cue-driven Gaussian injection into the leader channel (300 Hz peak)
dominates the recurrent drive toward the f(L) channel at every moment the
leader is on. There is no mechanism that lets a learned `L → f(L)`
transition *outbid* the leader's own ongoing cue during the leader epoch.

### Tang H-only forecast

As noted under Bug 3: direction state absent — Bug 3 blocks the Tang
forecast test even in isolation. We did not attempt to inject a synthetic
direction signal via `h_clamp` because (a) `h_ring.py` has no receptive
side for direction, (b) any synthetic direction input would bypass rather
than test the architecture. Test reported as structurally impossible.

## Verdict summary

| bug | status | primary evidence |
|---|---|---|
| 1. balanced all-pairs training | **CONFIRMED** | B1 empirical + stimulus.py L199-207 |
| 2. post-trailer MI gate        | **CONFIRMED** | stage_1_gate.py L161 + train.py L721-740; B4b all-seeds FAIL |
| 3. Tang no direction state     | **CONFIRMED** | h_ring.py grep-direction = 0; stimulus.py L315 |
| H-only forecast test (B5)      | **FORECAST ABSENT** | P(amax = f(L)) = 0.000 on biased schedule |

Sprint 5d's Case C (intrinsic V1) verdict stands. Case A ("H learning rule
wrong") was the wrong frame — the learning was never given a chance. The
upstream diagnosis is stronger: **schedule, gate, and architecture are all
bugs.**

## Recommended Sprint 5e-D scope

Fix in this order — the cheapest bug first, but all three are necessary:

1. **Schedule (cheap)**: replace
   `richter_crossover_training_schedule` with a biased deranged-permutation
   variant; keep the balanced 36-pair schedule as the *test/assay*
   generator (not the training one). Use reviewer's 80/20 split. ~40 LOC.

2. **Gate (cheap)**: add `check_h_preprobe_forecast_mi` to
   `validation/stage_1_gate.py` — MI(expected_next_idx,
   argmax(H[pre_trailer_100ms])) with the same 0.05-bit floor — and rewire
   `run_stage_1_hr` to compute per-trial argmax in the last 100 ms of the
   leader, not at `trailer_end + 500 ms`. ~60 LOC.

3. **Architecture (substantive)**: B5 shows schedule+gate alone are not
   sufficient. A single Wang ring cannot hold two simultaneous bumps (one
   "current sensory" tracking the input, one "predicted next" driven by
   learned recurrent weights). The minimal fix is a **context/prediction
   split**: two ring populations linked by a learned orientation→orientation
   transition map.
   - `H_context`: tracks current input (today's single ring; cue-driven).
   - `H_prediction`: receives a learned transform of H_context output and
     is **not** cue-driven during the leader epoch; its argmax during the
     pre-trailer window is the forecast.
   - Learned transform via delayed Hebbian/STDP on H_context → H_prediction
     using the biased schedule.
   Estimated scope: new `h_pred_ring.py`, new learned synapse group,
   re-plumb `run_stage_1_hr`, extend Stage-1 gate; Researcher + Coder
   pass. Rough guess ~800 LOC + Stage-1 retrain.

4. **Tang direction state** (when Tang is in scope): add a 2-channel
   direction input to H_T (or a 2D orientation×direction grid) before
   attempting any Tang forecast training.

If user wants a smaller first step, (1)+(2) alone are a prerequisite for
any architectural test; without them no architecture can be evaluated
against a forecast gate. But (3) is the only thing that can produce a
positive pre-probe forecast.

Budget used: B1 + B4a + B4b compute-free; B5 24.6 min wall. All well under
the 90 min cap.
