# Phase 4 Doctrine — Private Scratch Notes (NOT a methodology doc)

**Status**: SCRATCH. Not versioned. Not canonical. Do not cite this file.
**Purpose**: Unstructured thinking park for Validator doctrine on Phase 4 regime
replication, to be picked up and promoted to a real methodology doc at Phase 3 start.
**Created**: 2026-04-10
**Owner**: validator
**Promotion trigger**: Phase 3 start AND multi-seed rate runs completed.

---

## Why this is scratch, not a methodology doc

The firing-rate methodology drifted through four versions (v1→v4) because it was
written before primary sources were nailed down. I will not repeat that mistake
for Phase 4. These notes exist so I don't lose context between now (2026-04-10)
and Phase 3 start, but they are NOT to be cited or treated as a ruling.

Promotion rules: when Phase 3 starts AND the multi-seed script has executed AND
measured σ is available, use these notes as seed material for drafting
`validation/phase4_regime_methodology.md` as a proper v1 doc.

---

## Joint ruling 2026-04-10 (team lead + validator) — pinned

1. **Hard gates (8 of them)**: sign + rank gates. Endorsed as-is. No magnitude
   checks. Sign/rank is robust to single-seed σ because effect sizes dwarf noise.

2. **M10 soft gate — regime-aware split**:
   - a1 (dampening, central 0.70): ±20% → [0.56, 0.84]
   - a2a (transition, central 1.00): absolute ±0.10 → [0.90, 1.10]
   - e1 (sharpening, central 1.13): ±20% → [0.90, 1.36]
   - Rationale: dampening/sharpening phase boundary runs through M10=1.0.
     Uniform ±20% at a2a (→[0.80,1.20]) crosses both regimes — unacceptable.

3. **FWHM soft gate**: ±50% accepted, but PAIRED with a rank-order auxiliary
   gate: e1 narrow magnitude > a2a narrow magnitude > a1 widen magnitude.

4. **M7 soft gate**: ±40% accepted. Sign gate is what matters.

5. **Multi-seed execution is a HARD PRECONDITION for Phase 4 gate finalization**.
   Task #32 (multi-seed script) remains prep-only until Phase 3 start, when the
   team lead escalates it to prep+execute. Three additional seeds per config
   (total n=4) replace estimated σ with measured σ before any gate locks.

---

## Gate values from the Researcher's pack (Phase 4 targets, current as of 2026-04-10)

Source: `~/.claude/plans/snn_rate_model_targets.md`
Verified against `RESULTS.md` lines 251, 255, 257, 278 on 2026-04-10.

| Config | λ_sens | l4w | snn_w | l23w | M7 | M10 | ΔFWHM | FB contrib | Regime |
|---|---|---|---|---|---|---|---|---|---|
| a1 | 0.0 | 2.0 | 0.0 | 1.0 | −0.047 | 0.70 | −3.9° | +0.143 | WIDENS (dampening) |
| a2a | 0.12 | 2.0 | 0.0 | 1.0 | +0.028 | 1.00 | −3.2° | −0.001 | NARROWS (transition) |
| e1 | 0.3 | 2.0 | 0.0 | 3.0 | +0.104 | 1.13 | −11.6° | +0.002 | WIDENS (sharpening) |

Single-seed. seed=42. stage1=2000, stage2=5000.

---

## Open questions for when I promote this to a real doc

- **σ estimates**: researcher pack estimates σ_M7≈0.02, σ_M10≈0.10, σ_FWHM≈1°.
  These are from neighbouring-config variability, not true seed variance. Must
  be replaced with measured σ before locking.

- **SNN σ may exceed rate-model σ**: spiking networks are stochastic even at
  fixed seed due to integration discretization. Phase 4 gate lock should probably
  compare SNN central value ± SNN σ against rate-model central value ± rate σ,
  not against a scalar target. Need to think about whether to use z-scores
  (|Δ| / sqrt(σ_SNN² + σ_rate²) < threshold) or bounded overlap.

- **FWHM rank check — RESOLVED 2026-04-10**. My first-draft rank was
  "e1 narrows > a2a narrows > a1 widens", which conflated two different
  measurements (FB contribution sign vs ΔFWHM across training). Researcher
  correctly rejected the three-way rank because a1 vs a2a ΔFWHM gap is only
  0.7° (−3.9° vs −3.2°), well within estimated σ_FWHM≈1°, so it's not gatable.
  **H9 final form**: `FWHM Δ(e1) < FWHM Δ(a1)` AND `FWHM Δ(e1) < FWHM Δ(a2a)`
  — i.e., e1 is strictly the most-narrowed config, with ~3× gap above σ.
  This is the defensible subset. Revisit once multi-seed measured σ lands:
  if σ_FWHM turns out smaller than estimated, consider extending H9 to include
  a1 vs a2a strict ordering.

- **Multi-seed noise floor for hard gates**: even sign-based hard gates can flip
  at small effect sizes. Need to confirm that M7 effect sizes (0.047, 0.028, 0.104)
  exceed σ_M7 ≈ 0.02 by at least 2×. For a2a (0.028) this is barely 1.4× — may
  need to tighten that particular sign gate to "sign + non-zero magnitude threshold".

- **Config reproducibility check**: before gating SNN against these numbers, I
  should have the Coder/Researcher run the rate model at seed=42 and confirm
  the RESULTS.md numbers reproduce bit-exactly. If not, there's drift in the
  rate model itself and Phase 4 gates are built on sand.

---

## Policy principles to preserve

Lifted from firing_rate_methodology.md §4.5 (my own v4 commitment) — same
doctrine applies here:

1. **No round-number cleanup**. If measured σ gives tolerances like ±0.093,
   do not round to ±0.10 for readability. Round numbers invite drift.

2. **Every number cites a source**. For Phase 4, the "source" is the measured
   rate-model distribution from the multi-seed run, not an estimate.

3. **Provenance-visible revision log**. When a gate changes, record why, when,
   and what evidence changed. Same style as firing_rate_methodology.md.

4. **Inference chains disclosed**. If any gate uses an inferred value (e.g.,
   rate-to-SNN correction factor), disclose the chain in a dedicated §
   analogous to §4.5.

5. **No speculative gating**. If I don't have evidence, the gate stays WARN,
   not GO or NO-GO. Uncertainty routes to warning, not to false precision.

---

## Things I do NOT yet have opinions on

- How to handle the F1-F4 feedback regime metrics (from §13-14 of RESULTS.md)
  if the Researcher includes them in the Phase 4 pack later.
- How to gate training loss curves (sensory/fb loss at stage1/stage2 end) —
  probably a trainer-side concern, not a regime-replication concern.
- How to gate wall-clock or memory (the BPTT optimization pack from Task #25
  may constrain this but it's a throughput issue, not a correctness issue).

Revisit these when the real Phase 4 pack lands.

---

## Next action when I pick this up

1. Confirm multi-seed rate runs have been executed and σ values are measured.
   - Script: `scripts/run_rate_multiseed.sh` (prepared 2026-04-10, Task #32)
   - Configs: a1, a2a, e1; seeds: 1, 7, 13; output: `results/multiseed/`
   - Total n=4 per config (3 new + existing seed=42). **Flagged to Researcher
     that n=4 may be thin** — rule of thumb is n≥5 minimum for stable SE,
     n≥10 preferred. Team lead should rule on seed count at escalation time.
2. Replace the σ estimates in the Researcher's pack with real numbers.
3. Re-evaluate all soft gates: do they survive the new σ? Do any need tightening
   because measured σ is smaller than estimated? Do any need widening?
4. Re-evaluate the M7 sign gate for a2a: 0.028 / measured_σ_M7 ≥ 2.0?
5. **Revisit H9 FWHM rank check** — if measured σ_FWHM < 0.35° (half the
   a1/a2a gap of 0.7°), consider extending H9 from the 2-condition subset
   (e1 < a1 AND e1 < a2a) to the full 3-way rank (e1 < a1 < a2a or
   e1 < a2a < a1 — whichever the rate data shows with new σ).
6. Promote this scratch file to `validation/phase4_regime_methodology.md` v1
   with a proper revision log and provenance section.
7. Delete this scratch file once the real doc exists.
