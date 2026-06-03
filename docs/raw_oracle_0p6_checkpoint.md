# Raw L2/3 Activity Reliability Checkpoint

Date: 2026-06-03

Branch: `v1-l23-raw-oracle-0p6`

## Hard Target

The user requirement is `raw_oracle@5 >= 0.60` for exact L2/3 activity reliability. The older validator floor of `0.45` is not acceptable as success.

Strict validation must keep:

- `--l23-video-min-raw-oracle-at-k 0.60`
- `--require-event-driven-ff-plasticity`
- `--require-emergent-ff-gain`
- `--require-l23-activity-reliability`
- no future frames, labels, held-out frames, HVA feedback, manual eval gain, windowed count shortcut, or weight-only shortcut
- all existing biology gates for OSI, coverage, SOM/PV, recurrence, natural-video timing, sparsity, scaling, and sensory baselines

## Current Code State

Implemented locally:

- Event-driven local `L4E -> L23E` plasticity using pre/post traces.
- Old windowed spike-count coactivity rule disabled for accepted candidates.
- Old weight-only heterosynaptic competition disabled for accepted candidates.
- Event/no-cheat summary fields and edge audit CSVs.
- Final post-video orientation/coverage/SOM validation artifacts.
- Spike-recording segmentation workaround for large `L4E/L23E` recording buffers.
- Post-video inhibitory stabilization with local postsynaptic tail gate.
- Second-sweep inhibitory eta schedule.
- Env-gated pre-replay recurrent-only consolidation:
  - `V1_VIDEO_RECURRENT_ONLY_CONSOLIDATION_ENABLE`
  - `V1_VIDEO_RECURRENT_ONLY_CONSOLIDATION_PASSES`

Modified tracked files:

- `genn/v1TwoLayerModel.cc`
- `tools/validate_full_plasticity.py`
- `MEMORY.md`

## Best Strict Event-Driven Baseline So Far

Artifact prefix:

```text
/scratch/proj/v1_snn_l4_l23/genn_run_l23activity_postinhstab_tailgate_20260601T113500Z/genn/v1_l23activity_postinhstab_tailgate_sweeps2_eta1_eta014_h1p5e6_pt010_ffhomeo120_finalctx_full
```

Result:

```text
FAIL_COUNT=0 under previous 0.45 raw threshold
raw_oracle@5=0.484115
raw_ceiling_fraction=0.750505
L23E_repeat_corr=0.470615
frame_top1=0.712240
frame_top5=0.979167
final frac_lt1=0.853125
final p99=4.974218
final L23 OSI=0.795805
```

This is biologically cleaner than the earlier shortcut result, but it does not meet the user target `raw_oracle@5 >= 0.60`.

## Failed 0.60 Attempts After Branching

### Recurrent-only pre-replay consolidation, one pass

Artifact prefix:

```text
/scratch/proj/v1_snn_l4_l23/genn_run_l23activity_postinhstab_tailgate_20260601T113500Z/genn/v1_l23raw_recuronly1_tail014_h1p5e6_pt010_ffhomeo120_finalctx_full
```

Result:

```text
raw_oracle@5=0.483594
leaky_oracle@5=0.641146
raw_ceiling_fraction=0.754265
L23E_repeat_corr=0.487755
frame_top1=0.696615
frame_top5=0.964844
final frac_lt1=0.849375
final p99=4.897656
```

Conclusion:

- Correctly placed before replay.
- Too weak to matter: `l23ee_changed_frac=0.013705`, `mean_gain_ratio=1.000190`.
- Did not improve raw exact activity reliability.
- Debugger ruled out recurrence-only pass count 1 as an effective lever.

### Reduced event-trace heterosynaptic LTD

Candidate: `V1_VIDEO_FF_EVENT_TRACE_HETERO_MINUS=0.00000075`, recurrent-only consolidation off.

Artifact prefix:

```text
/scratch/proj/v1_snn_l4_l23/genn_run_l23activity_postinhstab_tailgate_20260601T113500Z/genn/v1_l23raw_h0p75e6_pt010_tail014_ffhomeo120_finalctx_full
```

Result:

```text
raw_oracle@5=0.473177
leaky_oracle@5=0.629688
raw_ceiling_fraction=0.751447
L23E_repeat_corr=0.424037
frame_top1=0.684896
frame_top5=0.976562
final frac_lt1=0.847500
final p99=4.844010
final OSI=0.793104
```

Conclusion:

- Event/no-cheat and biology gates mostly stayed intact.
- Raw exact reliability worsened.
- Final sparse fraction also failed.
- This should not be repeated as-is.

## Current Diagnostic State

The latest debugger finding before shutdown:

- `raw_oracle@5 >= 0.60` is not blocked purely by leaky repeat ceiling; current leaky ceiling is above `0.60`.
- The core blocker is exact top-k winner identity retention across repeats.
- Post-video inhibitory stabilization cannot affect replay raw metrics because it runs after replay.
- One recurrent-only pass was correctly placed but too weak.
- Reducing FF event-trace heterosynaptic LTD to `0.75e-6` worsened raw reliability.

Immediate next step after resuming:

1. Send the latest `h0.75e-6` failure to the debugger.
2. Diagnose why reducing `HeteroMinus` worsened raw exact reliability.
3. Choose one evidence-backed next mechanism.
4. Validate with `raw_oracle@5 >= 0.60`, not `0.45`.

Do not report success unless the strict 0.60 target and all biology/no-cheat gates pass.

## Active Run State

No active pod job needs to be preserved before shutdown. The latest candidate finished and failed.

