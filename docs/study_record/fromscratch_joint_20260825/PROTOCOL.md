# From-scratch joint-training probe (α=0.5, surround s=0.04) — protocol
Opened 2026-08-25 by team-lead. User order: "try it and report" — replace the two-stage
pretrain+arm protocol with dual training (task+energy objective) from step 0.

## Question
With the energy objective active from initialization — no pretrain stage, gains trainable from
step 0 — does the dampening regime (a) collapse (energy's trivial minimum wins before task
competence exists), (b) converge to generic untargeted suppression (energy satisfied without
prediction-referenced structure), or (c) self-sequence (task gradients dominate early, the
network builds the task-informative FB map first, then exploits it — landing at the two-stage
strategy)?

## Design (single cheap observational seed — n=1, no kill gates; the outcome is the finding)
- Harness: byte-copy of the exact frozen harness used for the s=0.04 ladder dampening runs;
  deltas exactly: (1) skip the pretrain stage — arm starts from fresh init; (2) parameter policy
  from step 0 = full set (cell + FB + gains); (3) 11000 joint steps at α=0.5 (budget-matched to
  3000 pretrain + 8000 arm); (4) numbered weight snapshots every 500 steps (trajectory analysis).
  Diff-audit against the source harness printed to RUN_LOG — nothing else may differ.
- Config: family cell s=0.04, σ=4.0, α=0.5, seed 8 (same init lineage as the two-stage run).
- Measurement: standard frozen assay on the endpoint (M, center, flank, H, vitality) against the
  two-stage seed-8 s=0.04 arm endpoint (M 0.2961) and the pretrain host (M 1.3554, decode 0.7578,
  k +0.5457). Per-snapshot trajectory: k (float64 softplus on circ_raw), FB orientation decode
  (transplant-study metric), mean rate — the ORDER of decode-rise vs k-sign-flip answers (c).
- Envelope: cuda:0, MemAvailable ≥25 GB before launch, sequential, PYTHONHASHSEED=0 python3 -B,
  writes only under /home/vishnu/scratch/fromscratch_joint_20260825/ and this study dir; all
  frozen roots READ-ONLY.

## Log
- 2026-08-25: Protocol opened; run + analysis dispatched to coder.
- 2026-08-25: COMPLETE — outcome (b)-with-structure, not (a), not (c). Run clean (exit 0, diff
  audit = hunks (a)+(d) only, (b) satisfied by existing arm policy, (c) via CLI steps; init
  bitwise == donor lineage, init k +0.5457). Endpoint (joint | two-stage | host): M 0.2487 |
  0.2961 | 1.3554; center 0.1411 | 0.1436 | 1.8070; flank 0.3879 | 0.4999 | 0.8102; H 0.1759 |
  0.1991 | 0.4769; decode_A−B −0.2018 | −0.0285 | +0.7578; decode_A 0.1766 | 0.2859 | 0.7603;
  k −3.2479 | −3.5016 | +0.5457; vitality PASS, untripped. ORDER READOUT: k crosses zero at step
  ~700; decode_A−B NEVER becomes task-informative (max +0.0007 @500, final −0.20) — the
  self-sequencing ingredient never occurs. Rate dip to 0.027 @1000 then partial recovery to
  0.0415. Center<flank asymmetry SURVIVES without an informative FB map (flank/center 2.75 vs
  two-stage 3.48) — measured fact, mechanism not established (open). n=1 seed 8, observational.
  Artifacts: /home/vishnu/scratch/fromscratch_joint_20260825/ (RUN_LOG diff audit,
  results_joint.json, 22 snapshots). Reported to user. PROBE CLOSED.
- 2026-08-25: α0.0 COUNTERPART (user follow-up) — sharpening phenotype does NOT appear from
  scratch. Endpoint (joint | two-stage | host): center 0.9020 | 1.1895 | 1.8070; flank 0.8549 |
  0.8279 | 0.8102; H 1.0000 | 0.9907 | 0.4769; M 0.8813 | 0.9672; decode_A−B +0.0475 | +0.3825 |
  +0.7578; k −0.0084 | +0.0473 | +0.5457. k trajectory +0.5457 → ≈0 by step ~900 and parks there
  — pure-task training with free gains switches the feedback loop OFF; placement perfect without
  it. Vitality PASS, clean run, same verified init lineage. n=1 seed 8.
  Artifacts: runs/joint_alpha0p0/, results_joint_alpha0p0.json. ANSWER TO USER: dampening
  emerges without two-stage (crude-substrate variant), sharpening does not.
