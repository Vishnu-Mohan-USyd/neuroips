# Expectation Energy Repair Validation

> **HISTORICAL / NONCANONICAL.** This note documents an earlier repair program
> and is not the source of truth for the current figures. Use the canonical
> [emergent task–energy axis guide](emergent_task_energy_axis.md) for the current
> model, minimal objective, protocol, results, and reproduction commands. The
> legacy repair below used expected-versus-orthogonal and floor-relative terms;
> it must not be presented as evidence for the current emergent objective.

## Legacy remainder

This note records the repaired expectation-suppression result for the SOM/VIP
models. Expected stimuli now use lower L2/3 activity than matched 90-degree
unexpected stimuli while preserving the intended precision tradeoff:
sharpen/attend has smaller energy savings with better decoding, and
dampen/save saves more energy with degraded expected precision.

Generated checkpoints and logs were external, developer-local artifacts and
are not committed. Their machine-specific directories are intentionally not
published. In the commands below, set these logical roots to regenerated copies:

- `INDEPENDENT_REPAIR_ROOT`: independent repair output directory;
- `COMBINED_REPAIR_ROOT`: combined Phase B repair output directory.

Canonical repository checkpoints were not overwritten.

## Validation Commands

Independent sharpen/dampen checkpoints:

```bash
python tools/validate_energy_decoding_shape.py \
  --sharpen "$INDEPENDENT_REPAIR_ROOT/ckpt_energy_repair_sharpen.pt" \
  --dampen "$INDEPENDENT_REPAIR_ROOT/ckpt_energy_repair_dampen.pt"
```

Combined context checkpoint:

```bash
python tools/validate_ctx_energy_decoding_shape.py \
  --ckpt "$COMBINED_REPAIR_ROOT/ckpt_ctx_energy_repair.pt"
```

Recorded validation artifact names were:

- `$INDEPENDENT_REPAIR_ROOT/validate_energy_decoding_shape_stage4.log`;
- `$COMBINED_REPAIR_ROOT/validate_ctx_energy_repair.log`;
- `$COMBINED_REPAIR_ROOT/validate_ctx_energy_repair.json`.

## Primary Validation Table

The deterministic validation set uses 216 expected-vs-orthogonal pairs:
36 starts crossed with velocities `[-3, -2, -1, 1, 2, 3]`, evaluated at
`K=4`. Primary energy is mean L2/3 activity.

| Model/state | Expected energy | Unexpected energy | Expected CE | Unexpected CE | Held acc | Shape |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Independent sharpen | 1.6351 | 1.7116 | 0.0215 | 0.4508 | 82.03% | pass, center enhanced |
| Independent dampen | 1.1112 | 1.3087 | 0.1872 | 0.0676 | 81.74% | pass, center suppressed below floor |
| Combined ctx=+1 attend | 1.7634 | 1.8175 | 0.0216 | 0.0737 | 81.48% | pass, center enhanced |
| Combined ctx=-1 save | 1.5535 | 1.5928 | 0.0560 | 0.0342 | 81.21% | pass, center suppressed below floor |

## Noisy Stress

At Gaussian noise `sigma=1.0`:

- Independent expected accuracy: sharpen 80.96%, dampen 64.01%.
- Combined expected accuracy: attend 90.7%, save 82.8%.
- Combined expected CE: attend 0.243, save 0.400.

These checks support the intended tradeoff: expected stimuli are cheaper in both
regimes, but attend/sharpen preserves more precise expected decoding than
save/dampen.

## Implementation Notes

The repair is objective-level rather than a hardcoded response profile:

- paired expected-vs-orthogonal energy contrast at `K=4`;
- current-orientation decoding pressure to prevent activity collapse;
- late/ramped prediction-derived metabolic pressure;
- dampen/save floor-relative expected-content suppression derived from the
  model prediction signal;
- Phase B context gating retained in one combined
  `SimpleNet(use_circuit=True, context=True)` checkpoint;
- Stage4 save-side hard-pair cleanup added symmetrically to attend cleanup.

The Phase B validator uses the built-in `net.decoder` contract. External
readouts are not used for pass criteria.

## Caveats

- The listed checkpoints are local generated repair artifacts, not committed
  canonical weights.
- The tools validate the specified paired-sequence regime and Phase B legacy
  equivalents; broader stimulus distributions should be validated separately.
- The implementation uses differentiable engineering objectives to approximate
  prediction-weighted metabolic pressure.
