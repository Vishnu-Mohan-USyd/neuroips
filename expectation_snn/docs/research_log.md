# Research log

## Phase 0 (2026-04-18)

- Branch `expectation-snn-v1h` created as orphan from `main` (no `v2-module-implementation` history).
- Skeleton tree per plan v5 sec 10.
- Single Brian2 env `expectation_snn` (Py 3.12, numpy >= 2.0, brian2 2.10.1).

## Tang 2023 paradigm (PDF-verified in Phase 0, 2026-04-18)

**Source**: Tang, M.F., Smout, C.A., Arabzadeh, E., Mattingley, J.B. "Expectation violations
enhance neuronal encoding of sensory information in mouse primary visual cortex."
*Nat Commun* 14:1196 (2023). DOI: 10.1038/s41467-023-36608-8.
PMID: 36864037. PMC: PMC9981605.

**Species**: male mice; two-photon calcium imaging (GCaMP6f); both awake and
anaesthetised mice.

### Stimulus paradigm — numbers verified from PMC full text

| Parameter | Plan v5 assumption | Paper value | Status |
|---|---|---|---|
| Orientation set | 6 at 30° steps, 0–150° | "uniform probability from 0 to 150° in 30° steps" | ✅ match |
| Per-item duration | 250 ms | "displayed for 250 ms" | ✅ match |
| Presentation rate | 4 Hz | "4 Hz presentation rate" | ✅ match |
| ISI | none | "no inter-stimulus blank interval" | ✅ match (back-to-back) |
| Rotation block length | 5–9 items before deviant | "rotated … for 5–9 presentations before jumping to an unexpected random orientation" | ✅ match (selection method in [5,9] not specified explicitly) |
| Deviant rate | ~12% | **implied** ~10–17% (1 deviant per block of 5–9; mean block ~7 → ~13%); not stated verbatim as a percentage | ≈ match, derive from block length |
| Rotation direction | n/a | clockwise OR anti-clockwise per block (e.g., 0→30→60 or 0→150→120) | **add to plan** |
| Random condition | n/a | orientations drawn pseudo-randomly uniform 0–150° in 30° steps | contrastive baseline |

### Additional parameters from Methods (for completeness)

- **Grating contrast**: 50 % (Michelson).
- **Spatial frequency**: 0.034 cycles/degree.
- **Stimulus extent**: full-screen gratings on monitor subtending 76.8° × 43.2°.
- **Trials per run**: 1800. Two runs each of Rotating and Random per session.

### Implications for H_T stimulus builder (brian2_model/stimulus.py)

- Sample rotation direction per block uniformly from {clockwise, anti-clockwise};
  block length uniformly from {5, 6, 7, 8, 9}; on block end, sample an unexpected
  orientation uniformly from the 5 remaining orientations (i.e., excluding the one
  the rotation would have produced next).
- No ISI between items.
- Per-item duration 250 ms ⇒ 4 Hz.
- At training scale for H_T: with mean block 7 + 1 deviant = 8 items per mini-sequence,
  a 1800-item run is ~225 mini-sequences. We can match Tang's within-session trial count.


## Pre-registration (committed before any assay runs)

TODO in Phase 0: after researcher's power calc (task #17) and validator's metric
sign-off (task #18), commit pre-registered:

- Balance sweep ladder S1..S5: r in {0.25, 0.50, 1.00, 2.00, 4.00}, g_total constant.
- Expected A1-A4 pattern per hypothesis H1-H4 (plan sec 6-7).
- Metric definitions (plan sec 5, with signatures from task #18).
- Seeds: main {42, 7, 123, 2024, 11}; held-out {99, 314}.
