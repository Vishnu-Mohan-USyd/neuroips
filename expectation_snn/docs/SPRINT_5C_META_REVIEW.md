# Sprint 5c Meta-Review — "H is not a prediction module"

Received 2026-04-20 after Sprint 5c dual-mode findings. External reviewer reads the branch + evidence and names the architectural failure:

> **The current null is not a profound biological null. It is a model-design null. H is not acting as a predictive memory state during the probe. It is mostly a sensory amplifier driven by the current V1 response. That makes the central experiment ill-posed.**

## Why this happened (reviewer's diagnosis)

### Flaw 1 — Stage-1 "prediction" gate measures post-trailer memory, not pre-trailer prediction
`check_h_transition_mi` and the training driver compare leader identity with H argmax **at +500 ms after trailer offset**. In a deterministic pair design, the actual trailer is correlated with the leader — so an H state that merely reacts to the trailer can pass the MI gate without ever predicting it. **Fake "prediction"** that dies under context_only testing.

### Flaw 2 — V1→H pathway makes H a sensory amplifier
Added in Sprint 5.5 because H rings were silent during measurement. Fixed the numerical problem but created the conceptual one: H is driven by current V1, not by learned context. H→V1 feedback becomes a same-stimulus recurrent loop, not a prior→V1 modulation.

### Flaw 3 — H persistence too weak in assay timing
NMDA recurrence exists, but measured H activity during Richter trailer / Tang next-item is 0 Hz when V1→H is disabled. The bump does not survive the inter-probe interval.

### Flaw 4 — Kok orientation-MVPA ceiling at 100% saturates the readout
2 orientations × 96 V1 E cells × 30 trials is trivially separable by linear SVM. Cannot detect any expectation-driven sharpening signal — assay is blind.

### Flaw 5 — Tang dominated by adaptation/SSA
After Δθ_prev stratification, rotational gain vanishes; raw contrasts are Δθ_prev-distribution artifacts.

## Four failure-case decision tree

Reviewer's decision tree based on diagnostic outputs:

- **Case A**: H has no pre-probe prior, but H-clamp works → **H learning/memory module is wrong**. Fix: split H into context + prediction populations.
- **Case B**: H has pre-probe prior, but H-clamp fails to produce route-specific V1 effects → **feedback interface is wrong**. Fix: apical modulation / SOM targeting / time windows.
- **Case C**: Effects survive even with H-off → **intrinsic V1 adaptation**. Fix: stronger covariate matching, shorter windows, mandatory H-off controls.
- **Case D**: Kok appears only at lower SNR → **decoder saturation was the confound**. Report Kok only in controlled non-saturated regime.

## Six diagnostics (Sprint 5d)

### D1 — Pre-probe prior index
Measure H channel rates in pre-probe window per assay. Compute `PI = H[expected_next] − H[current/leader]`. Kok: final 100 ms of cue-stim gap. Richter: final 100 ms of leader. Tang: final 50 ms of item t.
**Pass: PI > 0 (expected-next channel active before probe).**

### D2 — Forecast-vs-memory confusion matrix
Time-resolved confusion matrix: H argmax vs {leader, expected-next, current-sensory} across windows {early leader / late leader / pre-trailer / early probe / late probe}. For Richter, crucial window is **late leader**: does H decode expected trailer or just leader?

### D3 — Controlled H-clamp test (fastest-informative)
Externally drive the expected H channel before the probe; keep V1→H OFF during probe. 4 conditions × 3 time windows:
- no H feedback / direct-only / SOM-only / both
- 0–50 ms / 50–150 ms / 150–500 ms

Tests whether V1 feedback machinery is *capable* of sharpening (direct-only) or dampening (SOM-only), independent of H learning.

### D4 — Route impulse-response transfer function
Same H pulse, same V1 grating. Measure V1 E center/flanks, SOM, PV, adaptation, apical/SOM currents over 0–50 / 50–150 / 150–500 ms. Per route. Tests whether feedback produces transient excitation → sustained suppression (Kim 2022 LM→V1 pattern), which could explain why full-epoch integration washes out early sharpening.

### D5 — H-off adaptation baseline for Richter + Tang
Run corrected assays with all H→V1 feedback OFF and V1→H OFF. If step-structured Richter redistribution and Tang Δθ_prev effects survive, they are intrinsic V1 competition/adaptation, not expectation. Standard negative control.

### D6 — Kok SNR curve
Degrade the decoder via contrast reduction / input noise / cell subsampling (e.g., 8/16/32/64 E cells) / optionally 6 or 12 orientations. For each SNR level measure `Acc_valid − Acc_invalid`. Useful regime: baseline orientation decoding 55–80%.

## Structural fix (contingent on Case A verdict)

If D1/D2 show no pre-probe prior but D3 clamp works, the minimal fix is:

```
H_context  ->  H_prediction  ->  V1 feedback
```

- `H_context` holds cue / leader / current item.
- `H_prediction` represents expected upcoming feature.
- Only `H_prediction` projects to V1 apical + SOM.
- Learned `H_context → H_prediction` weights perform the predictive transform.

For Richter: leader → H_context → H_prediction(trailer) before trailer onset.
For Tang: current item + rotation direction → H_context → H_prediction(next item) before next item.
For Kok: cue → H_prediction directly.

## Sprint discipline

> Do **not** run a broad V1→H gain sweep yet (tunes the amplifier, doesn't test whether the network CAN implement expectation).
> Do **not** write a null-result paper until diagnostics localize the failure.
> Do **not** add more biological detail — the missing computational operation is specific and local.

The next sprint answers four questions: (1) is there a prior in H before the probe? (2) if not, can a clamped prior modulate V1 correctly? (3) are observed effects still present without H feedback? (4) is Kok unreadable because of decoder saturation?
