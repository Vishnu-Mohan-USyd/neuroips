# Research log

## Phase 0 (2026-04-18)

- Branch `expectation-snn-v1h` created as orphan from `main` (no `v2-module-implementation` history).
- Skeleton tree per plan v5 sec 10.
- Single Brian2 env `expectation_snn` (Py 3.12, numpy >= 2.0, brian2 2.10.1).

## Tang 2023 paradigm (PDF-verified in Phase 0)

Source: Tang et al., *Nat Commun* 14:1196 (2023). DOI: 10.1038/s41467-023-36608-8.

Verified timing numbers from the paper Methods (TBD — WebFetch result in-thread).

## Pre-registration (committed before any assay runs)

TODO in Phase 0: after researcher's power calc (task #17) and validator's metric
sign-off (task #18), commit pre-registered:

- Balance sweep ladder S1..S5: r in {0.25, 0.50, 1.00, 2.00, 4.00}, g_total constant.
- Expected A1-A4 pattern per hypothesis H1-H4 (plan sec 6-7).
- Metric definitions (plan sec 5, with signatures from task #18).
- Seeds: main {42, 7, 123, 2024, 11}; held-out {99, 314}.
