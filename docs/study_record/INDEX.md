# Study record — verbatim copies

Copies of the governing documents of the surround-inhibition study, byte-exact
as of packaging (2026-08-23). Originals live in the frozen study dir
`/home/vishnu/neuroips_analysis/flank_sharpening_20260819/` on reuben-ML.

| file | role | sha256 (at copy) |
|---|---|---|
| `PROTOCOL.md` | Lead's protocol: goal, pre-registered criteria, envelope, full decision log (all phases, rulings, amendments A1–A4, Phase-4/ladder pre-registrations) | `cf5035afd40a5453d109fc4ccae94d0f4c5ee439b866c94c40a743777f79ab50` |
| `DESIGN.md` | Researcher's design: candidate ranking, the exact 2-constant diff, kernel footprint table, six primary sources | `85286bba333ea081dd8ffe88cd5c1dcc2b0e672aa719759979a0070cdfca2f38` |
| `DIAGNOSTIC_REPORT.md` | Debugger: s=0.5 collapse root cause (magnitude overdose, blanket s·Σf/36) | `da191414298f90c729ef1597940d4ccf5f9d673e05815df750d6eec08565a41f` |
| `DIAGNOSTIC_REPORT_PHASE4_M.md` | Debugger: dampening M-shortfall decomposition (direct subtraction 64–74% / trained adaptation 26–36%, no regime change) | `549c4b71fa283435b448cc5e56161f6daefe226e0c9a375b4d7a959285b0e5a5` |
| `DIAGNOSTIC_REPORT_PHASE4_LADDER.md` | Debugger: s=0.04 cells verified real; non-monotone trained-M explained (settling scatter, path sensitivity) | `6bed6dea015079eb0768c66f8db1c67de2b20263f669f30361c1ea7cbb0c66f0` |
| `VERDICT.md` | Validator: GO on the s=0.05 sharpening claim (all four seeds, A/A bitwise); Addendum 2: family claim at s=0.04 NOT CONFIRMED (Outcome O2) | `f5640239144e5f294f3c4279d31bd3b3f0eaa8c6574ff559dfb3f6da7e706285` |

Reading order for the science: PROTOCOL → DESIGN → DIAGNOSTIC_REPORT →
VERDICT (main + addenda) → DIAGNOSTIC_REPORT_PHASE4_M →
DIAGNOSTIC_REPORT_PHASE4_LADDER. The narrative synthesis is
`../ARCHITECTURE_AND_SCIENCE.md`.

## Follow-up investigations (snapshot 2026-08-25)

### `transplant_surround_20260823/` — weight-set strategy study

Which trained weight sets (CELL / FB / GAINS) carry the sharpening and
dampening phenotypes at s=0.04, via splicing + fresh FB controls (94 table
cells, 89 nets, measurement-only). Originals:
`/home/vishnu/neuroips_analysis/transplant_surround_20260823/` (docs) and
`/home/vishnu/scratch/transplant_surround_20260823/` (tables). Scripts:
`../../src/transplant/`.

| file | role | sha256 (at copy) |
|---|---|---|
| `PROTOCOL.md` | Lead's protocol: goal, baseline map, provenance correction (original study had no fresh FB controls), phases, rulings | `1c49cf20f73c072a07d0ff03f680ade6bb1f9344b6d8e8b52faf226e456aa357` |
| `DESIGN.md` | Pre-registered design: conventions, matrix, ρ machinery/floors/bands, gate chain, deeper analyses, labeled predictions | `3438e4c323acdfeb5da70560b5048e9f2afdf76298677cb7e5100634b0aa3d10` |
| `VERDICT.md` | Validator: **GO** on the strategy map (independent end-to-end re-derivation of every load-bearing cell from raw donor checkpoints; E0 anchor bitwise 8/8) | `ee83b5ed62aaf40be7bfaffe6c518fea9be0f60b81b25418db75aed85e8507a2` |
| `TABLES.md` | Full rendered results: raw markers, ρ tables, classifications, prediction confrontation, Q1–Q4/H-C1/H-C2, s→0, trip census | `55f24fd3a82a6c64d3cb341c3c1cdae9a6c92d8ab1f773438e7a7a65fdaf03df` |

### `fromscratch_joint_20260825/` — from-scratch joint-training probes

What happens when the two-stage protocol (3000 task-only pretrain → 8000
joint arm) is replaced by dual training from step 0 (n=1 observational, seed
8, both regimes). Run file: `../../src/train_fromscratch_joint.py`.
Originals: `/home/vishnu/neuroips_analysis/fromscratch_joint_20260825/` and
`/home/vishnu/scratch/fromscratch_joint_20260825/`.

| file | role | sha256 (at copy) |
|---|---|---|
| `PROTOCOL.md` | Lead's protocol: question (collapse / generic suppression / self-sequencing), design, envelope | `19a27971ec8948a806f53dd1f1ca52094e7f7d0611fcf4db03ad7e9f8fbb2887` |
| `RUN_LOG.md` | Coder's run log: source-harness sha proof, full diff audit (the exact two hunks), launch commands, measured summaries | `1c82b9dab5eebd39c0c18c5d879477614d8e50e110989b0df00972015933dc25` |
| `results_joint.json` | α=0.5 run: trajectory (k / decode / rates / profile per 500-step snapshot), endpoint vs two-stage arm and host, event-log curves | `8d11965f79e723eaf6f93e04b12ac27e0bcdbc0da2d0dd89cec42f480ce1f3a9` |
| `results_joint_alpha0p0.json` | α=0.0 run: same measurements | `d462e0da2a0e348cdaa829e4a6435c363e4ec75f5068619d19aa9544de070979` |
