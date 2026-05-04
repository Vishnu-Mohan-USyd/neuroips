# Paradigm-sign two-mechanism explanation (R1+R2)

Phase 4-6 debugger investigation, 2026-05-04. Scope: R1+R2 reference checkpoint
(`results/simple_dual/emergent_seed42/checkpoint.pt`) only.

## Question

The R1+R2 17-row cross-decoder matrix produces different ex/unex sign signs
across paradigms: M3R and VCD-test3 give small dampening, HMS / HMS-T give
small-to-moderate dampening, paired-fork HMM C1 is roughly null. Why? Are these
the same mechanism with different magnitudes, or independent drivers?

## Answer: two independent mechanisms

The 8-paradigm matrix on R1+R2 separates cleanly into two causal classes once
V2 feedback is ablated (`feedback_scale=0`).

**Mechanism 1 — V2-feedback channel-resolved gain modulation.** When V2 feedback
is intact, it modulates L2/3 activity at the predicted-channel band. The polarity
is set by V2 confidence (`pi`): high pi gives a broadband-excitatory effect at
the predicted channel, low pi gives a suppressive effect. Both polarities
produce the same outcome direction — decoder accuracy lower on V2-predictable
trials. Confirmed causal on M3R and VCD-test3 (native + modified) and on the
HMM C1 paired-fork V2-predictable strata: ablating V2 flips Δ_decC sign or
drives it to zero. The polarity flip itself is reproducible — pi-matching the
HMM C1 pe≤5° subset to HMS-T's pi-Q75 selection flips channel-resolved
Δr_stimch from −0.043 to +0.135, matching HMS-T paired-fork qualifying's
+0.178 broadband-excitatory signature.

**Mechanism 2 — non-V2 stim-decodability bias.** The HMS / HMS-T trajectory
split (3-march "ex" vs march+jump≥75° "unex") selects unex trials whose probe
stim is intrinsically less decodable by Decoder C, independent of V2 feedback.
Origin: a small ~+0.004 L4 representation bias toward 3-march stims (~1%
relative). Amplifier: the L2/3 recurrent kernel `W_rec` (a learnable circular
Gaussian) propagates this small bias by ~25× into ΔL23_stimch through the
readout window. Zeroing `W_rec` reduces |ΔL23_stimch| by 57% across HMS / HMS-T
/ M3R and FLIPS Δ_decC sign on HMS / HMS-T (dampening → sharpening). PV
ablation (Phase 5 H_PV) and Decoder C distribution (Phase 5 H_DecC) were both
falsified as the amplifier; SOM is structurally inactive in this network when
V2 is ablated (no drive source) so it cannot be a source of Mech 2. On Mech 2
paradigms, V2 feedback actually OPPOSES the underlying dampening — V2 ablation
makes the dampening LARGER, not smaller.

The two mechanisms are independent: Mech 1 paradigms collapse to ~0 or
sign-flip when V2 is ablated; Mech 2 paradigms strengthen (more dampening) when
V2 is ablated.

## Per-paradigm verdict table

(Phase 4e single-run numbers, n=2560 HMM trials per condition. Δ_decC =
unex − ex Decoder-C accuracy. Slight numeric variation between Phase 4d and 4e
on shared rows is from batch-consumption-order differences; qualitative
direction matches.)

| Paradigm | Intact Δ_decC | V2-ablated Δ_decC | Mechanism |
|---|---:|---:|---|
| M3R native | −0.033 | +0.069 | **Mech 1** (V2 causal, sign-flip on ablation) |
| M3R modified | +0.089 | +0.082 | INCONCLUSIVE (intact is sharpening direction in this run) |
| HMS native | +0.001 | −0.143 | **Mech 2** (V2 sharpens, NON-V2 underlying) |
| HMS modified | −0.016 | −0.127 | **Mech 2** (V2 partially masks NON-V2 dampening) |
| HMS-T native | −0.095 | −0.192 | **Mech 2** (V2 partially counters NON-V2 dampening) |
| HMS-T modified | +0.110 | −0.165 | **Mech 2** (V2 was masking strong NON-V2 dampening) |
| VCD-test3 native | −0.066 | +0.032 | **Mech 1** (V2 causal, sign-flip on ablation) |
| VCD-test3 modified | −0.019 | +0.009 | **Mech 1** (V2 causal, small magnitude) |

## Circuit-level file:line refs

- **Mech 1 — pi-modulated V2-feedback gating**: `src/model/network.py:307-318`.
  `precision_gate = pi_pred_raw / pi_max ∈ [0, 1]`;
  `scaled_fb = feedback_signal * feedback_scale * precision_gate`. At high pi
  the gate ≈ 1 and feedback is large (broadband-excitatory); at low pi the
  gate attenuates feedback toward 0.
- **Mech 2 — L4→L23 amplifier**: `src/model/populations.py:206-238` builds the
  W_rec circular-Gaussian kernel from learnable `sigma_rec_raw` /
  `gain_rec_raw`. `src/model/populations.py:274` applies it
  (`rec = F.linear(r_l23_prev, W_rec)`) — the recurrence step that compounds
  the bias by ~25× across the readout window.
- V2 ablation lever (used throughout Phase 4-6):
  `src/model/network.py:303` (`pi_pred_eff = pi_pred_raw * self.feedback_scale`).
  Setting `feedback_scale = 0` zeroes both the V2 excitatory feedback and the
  SOM drive (which sources solely from V2-negative-rectified).

## Open items

- **~43% of ΔL23_stimch on HMS / HMS-T survives W_rec ablation.** Residual
  paths not yet tested: L23 softplus nonlinearity, L23 tau integration,
  feedforward L4→L23 W (currently identity buffer at `populations.py:204`).
- **Origin of the small L4 representation bias** (~+0.004 ΔL4_stimch on
  3-march vs rare-jump stims) is unexplored. Candidates: V1→L4 connectivity
  asymmetry, L4 adaptation × march-context interaction, contrast-response
  asymmetry on rare orientation transitions.

## Source reports (uncommitted, in `/tmp/`)

`paradigm_catalog.md`, `debug_paradigm_sign_mechanism_report_v2.md`,
`phase4_paradigm_sign_thorough_report.md`, `phase4d_r1r2_paradigm_gaps.md`,
`phase4e_remaining_paradigms.md`, `phase5_open_questions.md`,
`phase6_amplifier_locus.md`. Validator GO/NO-GO verdicts captured per phase.
Per-trial JSONs at `/tmp/h14d_hms_diag.json`, `/tmp/h14d_pf_hmst.json`,
`/tmp/h14e_remaining.json`, `/tmp/h15_q1_mech2_locus.json`,
`/tmp/h15_q2_polarity_flip.json`, `/tmp/h16_amplifier_locus.json`. Diagnostic
scripts that produced these are committed under `scripts/_h11*`, `scripts/_h14*`,
`scripts/_h15*`, `scripts/_h16*` in the same commit as this document.
