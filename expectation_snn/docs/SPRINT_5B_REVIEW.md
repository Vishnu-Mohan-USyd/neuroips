# Sprint 5b External Review (reviewer-style critique)

Received 2026-04-20. Scanned the Sprint 5b findings report and Sprint 4.5 / Sprint 5.5 commits on `expectation-snn-v1h`.

## Summary

- **H1 falsification at `g_total=1.0` accepted** — the direction of 0/13 regime-switches across r is clear.
- **Interpretation of positive findings (Richter "local gain dampening", Kok signature, etc.) is NOT yet safe** — several confounds compromise the claim.

## Core concerns

### C1 — V1→H fix may have changed causal interpretation
The assay-time V1→H afferent (commit `6ba542c`) was necessary to produce any non-zero H during measurement, but it risks turning H into a **same-stimulus amplifier** rather than a **prior-carrying module**. A true expectation effect should show H active *before* the probe/trailer, and H→V1 currents should be measurable at probe onset regardless of same-trial V1→H drive. Not yet tested.

### C2 — Richter assay has repetition/adaptation confound
Current design: expected = θ_L = θ_T (same orientation continuation); unexpected = θ_L = θ_T + π/2 (orthogonal). This is **not** Richter 2022's cross-over. In Richter, the critical property is the same trailing item can be expected or unexpected depending on the preceding leader. The current design conflates expectation with:
- Same-channel spike-frequency adaptation
- Local E/I fatigue
- V1→H→V1 recurrence strongest for same-channel trials
The "−1.25 Hz matched-channel suppression" observed could be any of these plus actual expectation.

### C3 — Kok SVM decodes the wrong thing
Current SVM decodes **valid vs invalid** cue label. Kok 2012's actual claim is that **orientation decoding** is better for expected stimuli than unexpected (same probe orientation, compared across validity). Current 0.742 is also confounded by class imbalance (180 valid vs 60 invalid → 75% majority-class floor is near the reported number).

### C4 — Tang result is opposite the empirical signature + missing baseline
Model: dev−exp at matched θ = −1.39 Hz (deviants suppress). Tang 2023: dev>exp at matched θ (gain enhancement). Also Tang's paradigm includes Random as a baseline condition, absent in our assay.

### C5 — Full-epoch integration may wash out time-resolved structure
Kim/Shen LM→V1 physiology: transient excitation followed by prolonged suppression under sustained drive. A 500 ms integration window may capture late dampening only; early sharpening could exist and be invisible in current metrics.

## Recommended Sprint 5c sequence (reviewer ordering)

1. **5c-1 — feedback-state audit (FIRST, before any more parameter sweeps)**. Per assay, per condition, save time-window-split (pre-probe / 0–50 ms / 50–150 ms / 150–500 ms):
   - H E rate per channel
   - V1→H current
   - H→V1 apical current
   - H→SOM current, SOM rate, SOM→E current
   - PV rate, PV→E current
   - V1 E rate
   Produce per-assay figure showing current flow in expected vs unexpected / valid vs invalid.

2. **5c-2 — context-only V1→H mode**. Runtime flag `with_v1_to_h ∈ {continuous, context_only, off}`. `context_only` = V1→H active during cue/leader, disabled during grating/trailer. Re-run Kok + Richter. If effects collapse under `context_only`, current effects are NOT true prior effects.

3. **5c-3 — true route extremes**. `g_direct=g_total, g_SOM=0` (direct-only), `g_direct=0, g_SOM=g_total` (SOM-only), both zero (off). Time-windowed metrics.

4. **5c-4 — assay corrections**:
   - **Richter**: replace same-orientation repetition. Two options: (a) context tokens that are H-only (no visual leader → no adaptation confound), or (b) leader orientations with matched-adaptation control (never expected = leader=trailer; counterbalance leader distances).
   - **Kok**: decode **orientation** (45° vs 135°) separately for valid and invalid, balanced trial counts. Report Δ accuracy.
   - **Tang**: add **Random** baseline condition. Report Expected / Unexpected / Random. Adaptation covariates on previous-item orientation distance.

Only after these should any broad `(g_total, r)` 2D sweep run.

## Honest interim frame for the paper
> "In a minimal V1–H spiking feedback circuit, varying the direct-apical/SOM balance at fixed total feedback does not switch the sign of expectation effects. The current circuit robustly produces local expected-stimulus suppression, with direct feedback increasing the magnitude of this suppression. Whether this reflects genuine learned expectation, repetition/adaptation, or recurrent sensory amplification is the next question."

Paper cannot yet claim "Richter-style local gain dampening" without passing:
- Context-only V1→H check (still dampening without same-trial V1→H)
- Non-repeat Richter assay (expected != orientation repetition)
- Route-specific ablation (direct-only vs SOM-only distinguishable)
