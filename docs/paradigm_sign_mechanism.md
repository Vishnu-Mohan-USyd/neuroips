# Paradigm-sign two-mechanism explanation (R1+R2)

Phase 4-7 debugger investigation, 2026-05-04 → 2026-05-05. Scope: R1+R2
reference checkpoint (`results/simple_dual/emergent_seed42/checkpoint.pt`) only.
Phase 7 (2026-05-05) cross-checks Phase 4-6 conclusions under Dec A readout
alongside the Dec C readout used in Phase 4-6.

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

(Phase 4e Δ_decC numbers, n=2560 HMM trials per condition. Δ_decC = unex − ex
Decoder-C accuracy. Slight numeric variation between Phase 4d, 4e, and Phase 7
re-runs on shared rows is from batch-consumption-order differences; qualitative
direction matches. Dec A verdicts sourced from Phase 7 re-run; full Dec A
numerical values in `/tmp/phase7_decA_crosscheck.md` §1.)

| Paradigm | Intact Δ_decC | V2-ablated Δ_decC | Mechanism (Dec C) | Mechanism (Dec A) |
|---|---:|---:|---|---|
| M3R native | −0.033 | +0.069 | **Mech 1** (V2 causal, sign-flip on ablation) | **Mech 2** — DISAGREES |
| M3R modified | +0.089 | +0.082 | **Mech 2** (weak; V2 contrib +0.009, both Δ positive) | **Mech 2** (V2 contrib +0.256, strong) |
| HMS native | +0.001 | −0.143 | **Mech 2** (V2 sharpens, NON-V2 underlying) | **Mech 2** |
| HMS modified | −0.016 | −0.127 | **Mech 2** (V2 partially masks NON-V2 dampening) | **Mech 2** |
| HMS-T native | −0.095 | −0.192 | **Mech 2** (V2 partially counters NON-V2 dampening) | **Mech 1** — DISAGREES |
| HMS-T modified | +0.110 | −0.165 | **Mech 2** (V2 was masking strong NON-V2 dampening) | **Mech 2** |
| VCD-test3 native | −0.066 | +0.032 | **Mech 1** (V2 causal, sign-flip on ablation) | **Mech 1** |
| VCD-test3 modified | −0.019 | +0.009 | **Mech 1** (V2 causal, small magnitude) | **Mech 2** — DISAGREES |

5/8 paradigms have agreeing Dec C / Dec A mechanism verdicts; 3/8 disagree
(M3R native, HMS-T native, VCD-test3 modified). See "Decoder-robustness caveat"
below.

## Decoder-robustness caveat (Phase 7)

Phase 7 (2026-05-05) re-ran §1 V2-ablation, §2 pi-polarity, §3 W_rec-ablation
under Dec A (ckpt-bundled, ‖W‖=146, joint-trained with the network during
Stage 1 + Stage 2) alongside Dec C (separately trained on synthetic bumps,
‖W‖=37). Headline: **the substantive neural mechanisms are decoder-robust,
but several decoder-level verdicts are decoder-dependent.**

**Decoder-robust (hold under both Dec C and Dec A).** The channel-resolved
findings are properties of `r_l23` itself, not of any decoder readout, and
reproduce identically: Δr_stimch and ΔL23_stim sign-flip with pi-matching
(Phase 5 H_pi); W_rec ablation reduces |ΔL23_stim| by ~57% across HMS / HMS-T
/ M3R (Phase 6 H_W_rec). The amplifier identification (W_rec is the load-bearing
L4→L23 amplifier) and the polarity-gating mechanism (pi modulates V2-feedback
magnitude via `precision_gate`) are decoder-independent neural-mechanism
findings.

**Decoder-dependent (Dec C and Dec A disagree).** (1) Per-paradigm Mech 1 vs
Mech 2 verdict on three paradigms: M3R native (Dec C: Mech 1; Dec A: Mech 2),
HMS-T native (Dec C: Mech 2; Dec A: Mech 1), VCD-test3 modified (Dec C: Mech 1;
Dec A: Mech 2). 5/8 paradigms agree. (2) The pi-polarity flip on Δ_decoder is
Dec-C-specific: pi-matching flips Dec C's Δ from −0.066 to +0.004, but Dec A
reads near-constant +0.40 across all V2-predictable subsets (`HMM C1 pe≤5°`,
`HMM C1 pe≤5° + pi-Q75`, `HMS-T pf qual`) — pi has no effect on Dec A's
decoder-level Δ. (3) The W_rec-ablation Δ_decoder sign-flip on HMS / HMS-T is
Dec-C-specific: W_rec ablation flips Dec C's Δ from dampening to sharpening
(HMS −0.021 → +0.092; HMS-T −0.010 → +0.181), but Dec A stays NEGATIVE and
gets MORE negative (HMS −0.230 → −0.396; HMS-T −0.132 → −0.380).

**Why the disagreement.** Dec A is joint-trained with the network during
Stage 1 + Stage 2 with V2 active; its weights are tuned to V2-aligned r_l23
features and saturate near +0.40 Δ on V2-predictable trials, becoming
insensitive to the small modulations Dec C resolves. Dec C is trained
post-hoc on synthetic bumps, has smaller weight norm (~37 vs 146), and reads
finer channel-level modulation. The two readouts disagree most where the
underlying r_l23 modulation is small or where Dec A's V2-tuned features
dominate over the channel-resolved suppression that Dec C is sensitive to.

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
- **Decoder-robust vs decoder-dependent split is itself open.** Which decoder
  readout best reflects "what V1 actually represents" is unsettled. Dec A
  (joint-trained, ‖W‖=146) reports very different per-paradigm verdicts than
  Dec C (post-hoc synthetic-bump-trained, ‖W‖=37) on 3/8 paradigms (M3R native,
  HMS-T native, VCD-test3 modified) and on the pi-polarity / W_rec-ablation
  decoder-level Δ flips. Channel-level neural mechanisms (Δr_stimch, ΔL23_stim,
  W_rec amplification, pi-modulated polarity) ARE decoder-robust; only the
  decoder-level Δ verdicts vary. Resolving which readout is canonical (or
  framing both as legitimate) is downstream of this report.

## Source reports (uncommitted, in `/tmp/`)

`paradigm_catalog.md`, `debug_paradigm_sign_mechanism_report_v2.md`,
`phase4_paradigm_sign_thorough_report.md`, `phase4d_r1r2_paradigm_gaps.md`,
`phase4e_remaining_paradigms.md`, `phase5_open_questions.md`,
`phase6_amplifier_locus.md`, `phase7_decA_crosscheck.md` (Phase 7 Dec A
re-run), `validator_phase7_verdict.md` (Phase 7 validator GO/NO-GO). Validator
GO/NO-GO verdicts captured per phase. Per-trial JSONs at
`/tmp/h14d_hms_diag.json`, `/tmp/h14d_pf_hmst.json`, `/tmp/h14e_remaining.json`,
`/tmp/h15_q1_mech2_locus.json`, `/tmp/h15_q2_polarity_flip.json`,
`/tmp/h16_amplifier_locus.json`, `/tmp/h17a_decA_obs_wrec.json`,
`/tmp/h17b_decA_pi_polarity.json`. Diagnostic scripts that produced these are
committed under `scripts/_h11*`, `scripts/_h14*`, `scripts/_h15*`,
`scripts/_h16*`, `scripts/_h17*`.
