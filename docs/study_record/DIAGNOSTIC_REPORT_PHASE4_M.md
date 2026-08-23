# DIAGNOSTIC REPORT — Phase-4 α=0.5 M-shortfall decomposition
Debugger, 2026-08-19. Commission: decompose the P3_M FAIL (M 0.2475 vs band [0.2823, 0.3819],
original 0.332062, −25.5% rel) into (i) direct subtraction at measurement time vs (ii) trained
adaptation; state whether a trained regime change is involved; predict M(s) for the pre-registered
joint ladder s ∈ {0.02, 0.03, 0.04}. Checkpoint measurement only — zero training runs used.

## Method / pipeline identity

Probe: `probes/probe_m_decomposition.py` (sha256 below), cuda:0, sequential, MemAvailable-gated.
Every cell was measured with the CODER'S OWN `measure()` imported verbatim from
`phase4_endpoint_eval.py` — zero convention drift by construction. E0 gates (all asserted <1e-12,
all passed): original cell reproduces the frozen-M anchor 0.3320623037521497 exactly (doubling as
this session's bit-check); new-weights cells reproduce the coder's official M 0.2474838521908762
and A4 M 0.3020312939948634 exactly. t0 mean rate is BITWISE identical across weight sets
(0.16667840538150916 == 0.16667840538150916), confirming the shortfall lives entirely in adapted
activity. Budget cells additionally re-validated my reimplementation bitwise against the frozen
lib on both α=0.5 nets (max abs diff 0.0 on preds and rates) before any budget number was read.

## The closed 2×2 (M_auc_ratio units; weights × inference-s)

|  | s=0 at inference | s=0.05 at inference |
|---|---|---|
| **original frozen weights** | 0.3320623038 (anchor) | 0.2696719546 (mirror, NEW) |
| **new retrained weights** | 0.3020312940 (=A4) | 0.2474838522 (=official) |

Decomposition of the total gap 0.0845784516:

| component | value | share of gap |
|---|---|---|
| direct subtraction, on new weights (Mn0−Mn5) | 0.0545474418 | 64.5% |
| direct subtraction, on original weights (Mo0−Mo5) | 0.0623903492 | 73.8% |
| trained adaptation, at s=0 (Mo0−Mn0) | 0.0300310098 | 35.5% |
| trained adaptation, at s=0.05 (Mo5−Mn5) | 0.0221881024 | 26.2% |
| interaction (direct_new − direct_orig) | −0.0078429074 | 9.3% magnitude |

The lead's suggested split (~0.055 direct / ~0.030 adaptation) is VERIFIED at its margins: 0.0545
(direct, new-weights margin) and 0.0300 (adaptation, s=0 margin). The mirror closes the bracket:
depending on which margin you attribute the negative interaction to, direct carries 0.0545–0.0624
(64–74%) and adaptation 0.0222–0.0300 (26–36%). The interaction's sign means the RETRAINED
weights lose LESS to the surround than the original weights do — training partially adapted the
circuit to withstand the subtraction (part of what "adaptation" bought).

## Answer to the mechanism question

**The shortfall is NOT purely measurement-time arithmetic, but the arithmetic is dominant
(64–74%), and the remaining trained component is NOT a regime change:**

- Direct term = the proven blanket mechanism at benign dose. Measured mean subtraction is exactly
  s·Σf/36: 0.034678 = 0.034678 (official cell, Σf=24.970) and 0.034404 = 0.034404 (mirror cell,
  Σf=24.771) — the same identity proven in the root-cause report, at the α=0.5 operating point
  (Σf ≈ 25, same scale as α=0). The realized direct M loss (0.0545/0.0624 → 0.00909/0.0104 in
  raw rate units) is ~26–30% of the naive blanket because 79.4–79.6% of channels sit at the relu
  floor and can only lose their small positive part (same floor economics as the s=0.5 collapse,
  now benign). Dose response is near-linear beyond the first step (≈ −0.011 M per +0.01 s on both
  weight sets).
- Adaptation term = deepening of the SAME dampening phenotype, not a regime change. At s=0, the
  retrained net vs the original: aligned profile shape Pearson r = 0.970 (official-vs-original
  0.980); topology preserved (center 0.1043 < flank 0.5376); center-first suppression STRONGER
  (center 0.1043 vs 0.1496 — the P1 direction, exceeded); H BETTER at every inference-s
  (0.208–0.232 vs 0.167–0.194); M at s=0 (0.3020) is INSIDE the P3 band. Nothing task-relevant
  degraded; the trained state is a lower-activity point of the same regime.

WHY training settled ~0.030 lower at s=0 (energy-objective path vs other) was not causally
decomposed — not needed for this commission and would require training-dynamics probes; flagged
under unknowns.

## PREDICTION (clearly labeled; the ladder itself is the coder's)

Model: M_trained(s) = M_orig_bolt-on(s) − adapt(s), with the bolt-on curve MEASURED on the
original weights (0.30286 / 0.29137 / 0.28053 at s = 0.02 / 0.03 / 0.04) and adapt(s) scaled
linearly between the anchors adapt(0)=0 and adapt(0.05)=0.0221881 (s-margin; self-consistent:
reproduces the trained s=0.05 point exactly). Bracket: zero-adaptation ceiling and
constant-adaptation (0.0300) floor. P3_M bar: M ≥ 0.282253.

| s | central prediction | zero-adapt ceiling | const-adapt floor | predicted P3_M |
|---|---|---|---|---|
| 0.02 | 0.2940 | 0.3029 | 0.2728 | **PASS** (fails only under worst-case adaptation) |
| 0.03 | 0.2781 | 0.2914 | 0.2613 | **marginal FAIL** (−0.004 below bar centrally) |
| 0.04 | 0.2628 | 0.2805 | 0.2505 | **FAIL robustly** — even the ZERO-adaptation bolt-on (0.2805) is below the bar |

Sharp consequence: s=0.04 cannot pass P3_M even if training adapts not at all — the pure
measurement-time subtraction already exceeds the band allowance. The dampening-arm M bar is
monotonically pulled DOWN in s while the sharpening-arm flank bar (≤0.85) pulls UP in s; the
sharpening side is NOT predictable from bolt-on arithmetic (training amplifies flank suppression:
trained 0.789 vs s-switched-off 0.972 at s=0.05), so the joint window — if it exists — is
predicted to sit near s ≈ 0.02–0.03, with genuine risk that no s passes both regimes' bars.

## Verdicts

| hypothesis | verdict | evidence |
|---|---|---|
| Hm1 direct-subtraction dominant, phenotype intact | **CONFIRMED** | 64–74% of gap at both 2×2 margins; profile r 0.97, topology preserved, H improved, (new,s=0) M in-band |
| Hm2 trained regime change dominant | **RULED OUT** | adaptation is the 26–36% minority and is a regime-CONSISTENT deepening (stronger center-first suppression, better H); no qualitative departure |
| Hm3 direct term = proven blanket arithmetic | **CONFIRMED** | s·Σf/36 identity exact at both cells (0.034678/0.034404); realized loss = 26–30% of blanket via the 79–80% relu-floor share; near-linear dose curves |

## Unknowns

- The causal path of the 0.030 adaptation term (energy objective vs other training pressure) —
  unneeded here, would need training-dynamics probes.
- Sharpening-arm bar positions at s < 0.05 (adaptation-amplified; bolt-on cannot predict them).

Artifacts: `probes/probe_m_decomposition.py` + `probe_m_decomposition_report.json`
(full grid incl. s ∈ {0.01..0.04} cells, curves, budgets). All numbers trace to commands run
this session.
