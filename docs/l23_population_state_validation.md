# L2/3 Population-State Reliability Validator

`tools/validate_l23_population_state.py` is a standalone validator for repeated
natural-video L2/3 population-state exports. It is intentionally separate from
the full plasticity validator and does not change model behavior.

## Required Input

The primary input is an existing video site-rate CSV:

```text
repeat_index,frame_index,population,site_id,rate_hz
```

The tool accepts:

- A direct `*_video_site_rates.csv` path.
- A run directory containing exactly one `*_video_site_rates.csv`.
- An artifact prefix without the `_video_site_rates.csv` suffix.
- Multiple input paths, reported independently.

For 40x40 validation runs that use the central 32x32 core, pass:

```bash
--sheet-side 40 --core-side 32
```

For Fano-factor metrics, the tool also needs either:

- A complete `spike_count` column in the site-rate CSV, or
- A matching `*_video_frame_summary.csv` with:

```text
repeat_index,frame_index,frame_start_ms,frame_end_ms
```

If this duration/count information is absent, only the Fano/quenching metric is
reported as missing. Other repeat-state metrics still run.

## Metrics

- Population-vector repeat correlation:
  per-frame Pearson correlation of site-rate vectors across repeat pairs, plus
  flattened repeat-pair correlation and cosine similarity summaries.
- Controls/shuffles:
  deterministic frame-shuffle and site-shuffle controls using `--seed` and
  `--shuffle-count`.
- Odd/even RSM correlation:
  alternating repeat splits are averaged into even-rank and odd-rank templates.
  The frame-by-frame representational similarity matrices are correlated on the
  upper triangle.
- Held-out decoder:
  leave-one-repeat-out nearest-template frame decoding with cosine similarity.
  Reports top-1, top-k, rank, same-frame similarity, different-frame similarity,
  and shuffled-template controls.
- Fano/variability summary:
  across-repeat Fano factors from spike counts, or from `rate_hz * frame_duration`
  when a frame-summary file is available. The early/late summary compares the
  first and last frame windows; it is a variability-change diagnostic, not a
  biological pass criterion unless the run design makes those windows meaningful.

## Thresholds

No default biological pass thresholds are assumed. Optional thresholds can be
provided explicitly:

```bash
python tools/validate_l23_population_state.py \
  .runs/example/example_video_site_rates.csv \
  --threshold repeat_vector_corr_mean>=0.5 \
  --threshold odd_even_rsm_corr>=0.25
```

`metric=value` is treated as `metric>=value`. A threshold failure returns exit
code 1. Missing or invalid input schema returns exit code 2.

For the current pre-feedback V1 lower-circuit validation, use explicit gates
that treat population state as the biological target:

```bash
--threshold repeat_count>=4 \
--threshold frame_count>=64 \
--threshold repeat_vector_corr_mean>=0.45 \
--threshold repeat_frame_shuffle_gap_mean>=0.20 \
--threshold repeat_site_shuffle_gap_mean>=0.15 \
--threshold odd_even_rsm_corr>=0.50 \
--threshold odd_even_rsm_frame_shuffle_gap_mean>=0.15 \
--threshold odd_even_rsm_site_shuffle_gap_mean>=0.15 \
--threshold heldout_decoder_top1_accuracy>=0.20 \
--threshold heldout_decoder_top5_accuracy>=0.55
```

Raw exact top-k tile repeatability should remain a diagnostic/stress test, not
the primary biology gate for downstream prediction.

## Examples

Direct CSV:

```bash
python tools/validate_l23_population_state.py \
  .runs/v1_run/v1_run_video_site_rates.csv \
  --population l23e --seed 17 --shuffle-count 200
```

Run directory plus prefix:

```bash
python tools/validate_l23_population_state.py \
  .runs/v1_run --prefix v1_run --population l23e --json
```

Artifact prefix:

```bash
python tools/validate_l23_population_state.py \
  .runs/v1_run/v1_run --population l23e --output-json .agents/out/l23_state_metrics.json
```

## Limitations

- The validator does not infer repeats if `repeat_index` is missing.
- Missing site rows are treated as an input error, not filled with zeros.
- The held-out decoder is a simple nearest-template readout, chosen because it
  needs no extra dependencies and uses only held-out repeats.
- Fano factors require repeat-level counts or frame durations. Without those,
  the tool reports the exact missing file/columns instead of substituting a
  rate-only proxy.
- The early/late Fano delta is descriptive. It should not be described as
  stimulus-onset variability quenching unless the artifact contains an explicit
  baseline/stimulus window design supporting that interpretation.
