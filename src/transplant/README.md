# Transplant study scripts (weight-set strategy investigation)

Byte-exact copies of the scripts that executed Phase 2 of the surround
transplant study (2026-08-23/24), from
`/home/vishnu/scratch/transplant_surround_20260823/scripts/` on reuben-ML.
`ncommon.py` is the shared module every `n*` script imports (paths, cell
naming, sha-pinned anchor loading, the verbatim `measure()` core, the
instrumented-unroll clone); the numbered scripts are the execution order:

| script | phase |
|---|---|
| `n0_manifest.py` | G6 donor manifest (sha256 of all 48 donor files, before any build) |
| `n1_partition.py` | G1 partition proof + G2 pretrain equality |
| `n2_build.py` | 60 core hybrid nets + G0/G3/G5/EC1 gates |
| `n3_controls.py` | fresh FB controls R/N/Q (pinned RNG) + construction gates + null-edit gate |
| `n4_assay_all.py` | all endpoints on all 94 table cells (89 nets), determinism repeat |
| `n5_s0.py` | s→0 inference counterfactuals (floor-aware selection; correction disclosed in-file) |
| `n6_analyses.py` | per-set deltas/SVD, FB alignment geometry, gains/k tables |
| `n7_synth.py` | ρ machinery, pre-registered classifications, question/hypothesis verdicts, G6 re-verify |
| `n8_tables.py` | readable TABLES.md render |

Every measurement convention (cell IDs, ρ floors and bands, chimera gate,
gate chain, registered questions) is pre-registered in the study record:
`../../docs/study_record/transplant_surround_20260823/DESIGN.md`; the
protocol, validator verdict, and rendered results tables sit alongside it.
The scripts reference absolute reuben-ML paths (frozen donor checkpoints,
the original transplant-20260818 harness `tcommon.py`, the frozen assay
tools) and are archived here for record/review, not for standalone
execution.
