# Sprint 5d Post-Verdict Review — "The failure is upstream of learning"

Received 2026-04-21 after Sprint 5d's Case C + Case A verdict committed at `5d62550`. Reviewer argues the verdict is correct but **incomplete**: Sprint 5d diagnosed that H has no forecast, but didn't diagnose **why**. Three specific upstream bugs would make Case A's "learning rule wrong" the *wrong* story — learning never had a chance.

## Three reviewer-claimed bugs (to be verified with evidence)

### Bug 1 — Richter training schedule is balanced all-pairs (zero predictive contingency)

Claim: `richter_crossover_training_schedule` produces all 36 leader×trailer pairs with equal probability. `P(T | L) = P(T) = 1/6` for every L. There is no statistical regularity to learn. A cross-over test matrix is correct for the **test** phase; a uniform training schedule means the Stage-1 learning rule was never given anything to learn.

Reviewer's proposed corrected design: biased training with deranged permutation `f(L)`, e.g. `P(L → f(L)) = 0.80`, remaining `0.20` distributed across other trailers. Balanced cross-over only during testing.

### Bug 2 — Stage-1 MI gate measures post-trailer H (memory), not pre-trailer H (forecast)

Claim: `check_h_transition_mi` computes H argmax at `+500 ms after trailer offset`, and `run_stage_1_hr` implements the same. The measurement window is AFTER the trailer was presented, so an H that merely reacts to the actual trailer can pass the gate. A real forecast gate must measure H **before** trailer onset (late leader / pre-trailer window).

Reviewer's proposed corrected gate:
- `argmax(H[last_100_ms_of_leader]) == expected_trailer_channel`
- No post-trailer argmax allowed.

### Bug 3 — Tang has no direction state (underdetermined forecast)

Claim: Tang rotation blocks sample CW or CCW per-block; current orientation is the same for +30° and −30° next-item expectations depending on block direction. H_T is a single 12-channel orientation ring with no direction variable. The expected next orientation is formally underdetermined given only the current orientation state — H_T cannot do the task.

Reviewer's proposed fix: add a direction state (either separate direction channels or a 2D H_ctx grid of orientation × direction).

## Upstream-of-learning implications

If any of these three bugs is confirmed:
- Bug 1 → H_R never had learnable statistics → D1/D2 Richter failure is forced, not informative about learning rule adequacy.
- Bug 2 → Stage-1 gate is non-diagnostic; training could have produced anything and the gate would pass. Case A is confirmed regardless of H learning rule.
- Bug 3 → H_T failure is structurally guaranteed, not a failure of learning.

In that case:
- Case C (intrinsic V1 dominates) is **still valid** (D5 retention ≥ 0.70 is independent of H's state).
- Case A (H learning wrong) is **incomplete**: the learning was given no chance. The real upstream diagnosis is "schedule + gate + architecture inadequate."
- Sprint 5e cannot just "audit Stage-1 training" — it must fix schedule, gate, and architecture as a coupled set.

## What Sprint 5d did NOT measure

- `P(T | L)` computed directly from the training schedule output.
- H argmax at the **pre-trailer / late-leader window** (D2 measured pre-probe but interpretation assumed post-trailer was the previously-gated proxy).
- Whether an H-only forecast unit test (H ring alone, biased training, no V1, no assays) can forecast at all.

## Reviewer's Sprint 5e-Diag roadmap (diagnosis-only, no fixes)

1. **5e-A — Schedule + gate validators**
   - `validate_richter_training_statistics.py`: compute `P(T|L)` from schedule; assert `max_T P(T|L) ≥ 0.70` and `entropy(P(T|L)) < log(6) - margin`. Current schedule MUST fail this gate (expected).
   - `validate_tang_predictability_requires_direction.py`: prove same current orientation implies different next orientations under CW vs CCW; without direction state, forecast is formally ill-defined.
   - `validate_stage1_preprobe_forecast.py`: replace post-trailer MI with pre-trailer MI gate. Current Stage-1 network MUST fail this gate under biased schedule (the forecast isn't there to be measured).

2. **5e-B — H-only forecast unit test**
   - New script `scripts/diag_h_only_forecast.py`. No V1, no H→V1, no assays. Just H_R or H_T ring.
   - Train on corrected biased Richter schedule (80/20 split via deranged permutation).
   - Train on Tang with direction state (if feasible with current H_ring.py; else report failure mode).
   - Measure pre-probe H argmax.
   - If biased schedule + pre-trailer gate shows forecast is still absent: architecture fix (context + prediction split) is needed.
   - If biased schedule + pre-trailer gate shows forecast emerges: architecture may be OK, schedule was the entire bug.

## Sprint sequence (no fixes until all 5 deliverables land)

Only after all 5 of the following are committed with conclusive evidence:
1. Bug 1 code citation + P(T|L) measurement on current schedule.
2. Bug 2 code citation (stage_1_gate.py timing + train.py MI call site).
3. Bug 3 code citation (H_T has no direction variable; Tang sequence has direction degeneracy).
4. Schedule validator output.
5. H-only forecast unit test at biased schedule + pre-trailer gate.

Then Sprint 5e-D (architecture fix) dispatches. Not before.

## Honest interim status for paper

Sprint 5d verdict stands:
- Sprint 5b/5c "positive metrics" are ≥ 70% intrinsic V1 adaptation. H contributes ~0–30% depending on metric and can interfere.
- H does not carry a pre-probe prior.

Sprint 5d verdict DOES NOT yet say:
- Whether the H module can ever carry a forecast under the current architecture given a corrected training schedule.
- Whether the architecture needs a `H_context + H_prediction` split or just a schedule + gate repair.

Those are the questions Sprint 5e-Diag must answer BEFORE any architectural fix.
