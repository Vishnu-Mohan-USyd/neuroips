# Task #74 Handoff — V2 Laminar Predictive-Coding Circuit

**Branch:** `v2-model-build`
**Working directory:** `/mnt/c/Users/User/codingproj/freshstart_backup_2026-04-18`
**Date of handoff:** 2026-04-22
**Author:** coder2 (agent), composed at Lead's request mid-ablation
**Read-time target:** ~30 minutes to resume effectively

This document is written so that any incoming agent — Lead, Coder, Debugger, Researcher, or Validator — can pick this work up cold. Every section is load-bearing. The doc is organised top-down: scientific goal → architecture → fixes → validation state → residual failure → history → plan forward → preferences → artefacts → config → how-to-run.

---

## 1. Scientific goal

The V2 model is a minimal mechanistic laminar predictive-coding circuit built to adjudicate between two well-known but conflicting accounts of how expectation modulates sensory cortex. Kok et al. 2012 (Neuron) reports *sharpening* — expected stimuli produce higher decoding accuracy, narrower tuning, higher peak at the stimulus channel, and lower net activity. Richter et al. 2018/2019 (J Neurosci / J Cog Neurosci) reports *dampening* — expected stimuli produce lower activity with preserved tuning shape and preserved decoding. A biologically grounded network that can demonstrate one outcome should rule out the other given the same circuit.

The design choice is a laminar E/I microcircuit (L4 → L2/3E/PV/SOM → H area with E/PV and a context-memory module C) trained in two phases. Phase-2 does pure-local predictive activity-based training on a procedural world with hidden regimes — no task, no cue, no teacher. This yields a substrate that has learnt stimulus statistics via Urbanczik–Senn, Vogels iSTDP, homeostasis, and energy shrinkage. Phase-3 layers a task-specific readout (Kok: cue → delay → probe; Richter: leader → trailer bigrams) that installs expectation via a three-factor rule on the memory-to-apical task weights. The v4 plan (saved at `~/.claude/plans/come-to-me-with-streamed-grove.md` on an earlier run) approved this decomposition; the ongoing research log at `docs/research_log.md` captures chronology across related sibling branches.

**Non-negotiable constraint from the user.** See `memory/feedback_never_propose_null.md`. A null result — "the network learns neither sharpening nor dampening" — is not an acceptable terminus. It must demonstrate one or the other. Proposing "accept the null" is explicitly a rule-violation: the answer is always to dispatch the next diagnostic / ablation / fix. The canonical evaluation triad is also rule-bound (see `memory/feedback_expectation_metric_triad.md`): decoding accuracy + net activity + response-shape metrics (peak, FWHM, preference-rank suppression), all on expected-vs-unexpected, are required to disambiguate sharpening from dampening.

---

## 2. Architecture overview

The architecture is a three-stage laminar circuit plus a context-memory module and a prediction head. It is implemented entirely in `src/v2_model/` and composed in `V2Network` (`src/v2_model/network.py`).

**Populations, sizes, roles.** L4 E (128 units) is the frozen sensory front-end: a DoG+biphasic LGN bank feeding an 8-orientation Gabor energy model with 4×4 retinotopy, giving 128 L4 E and 16 L4 PV units. It has zero `nn.Parameter` entries and never changes during training; its SHA is asserted at checkpoint boundaries. L2/3 comprises three populations: L23E (256 units, pyramidal with apical and basal compartments), L23PV (16 units, fast inhibitory), and L23SOM (32 units, fast inhibitory, somatically dendrite-targeting). H is a downstream area with HE (64 units) and HPV (8 units). ContextMemory C carries a memory state `m` of size `n_c_bias=48` and exposes generic weights `W_hm_gen`, `W_mm_gen`, `W_mh_gen` plus task-specific splits `W_qm_task`, `W_lm_task`, `W_mh_task_exc`, `W_mh_task_inh`. The PredictionHead consumes (h_rate, c_bias, l23_apical_summary) and emits `x̂_{t+1}` for comparison against the next-step L4 E rate.

**Key pathways with init state.** `W_l4_l23_raw` has `init_mean=1.5` under Fix K, combined with a sparse orientation-biased top-k mask (see `scripts/v2/level_1_lgn_l4.py` for the probe that motivated Fix K). `W_rec_raw` on L23E has `init_mean=-5.0` with a like-to-like sparse recurrence mask. `W_pv_l23_raw` is `-6.5` post Fix L2 (PV divisive normalisation onto L23E, biologically this is the subtractive-inhibition channel carrying the Vogels ρ target). `W_som_l23_raw` is `-5.0` (init), `W_fb_apical_raw` is `-5.0`. L23SOM's `W_l23_som_raw` is `-4.2` post Fix M and `W_fb_som_raw` is `-5.0`. HE's `W_l23_h_raw` is `-5.5` post Fix N, `W_rec_raw` is `-5.0`, `W_pv_h_raw` is `-5.0`. The prediction head's four plastic weights (`W_pred_H_raw`, `W_pred_C_raw`, `W_pred_apical_raw`, `b_pred_raw`) are all at `-8.0` post Fix O (down from the earlier `-10.0` calibration that fell outside the plasticity clamp bounds).

**Plasticity active in Phase-2.** Urbanczik–Senn predictive-Hebbian fires on L2/3 pyramidal weights (`W_l4_l23_raw`, `W_rec_raw`, `W_fb_apical_raw`) with `ε = apical_drive − basal_drive`, and on the prediction-head weights with `ε = r_l4_{t+1} − x̂_{t+1}`. Vogels iSTDP fires on all inhibitory-to-E synapses with per-pop targets: `vogels_target_l23e_hz=3.0` on L23 I→E (`W_pv_l23_raw`, `W_som_l23_raw`), `vogels_target_h_hz=0.1` on H I→E (`W_pv_h_raw`), and `target_rate_hz=1.0` on I-pop self-regulation (`l23_pv.W_pre_raw`, `h_pv.W_pre_raw`). Hebbian outer-product (Vogels with `target=0`) fires on H E→E (`W_l23_h_raw`, `W_rec_raw`) and the context-memory generic weights (`W_hm_gen`, `W_mm_gen`, `W_mh_gen`). ThresholdHomeostasis updates θ on L23E and HE only (Fix A). EnergyPenalty applies L1 rate shrinkage to firing E units and L2 weight-shrinkage anchored at each weight's init under Fix P (`Δw = −β · mean_b(pre²) · (w − raw_prior)` in implicit-Euler form). All plastic updates are additive with a per-step clamp of `±0.01` and a raw-weight clamp of `[−8, 8]`. SOM excitatory inputs `W_l23_som_raw` and `W_fb_som_raw` are frozen in Phase-2 per Fix D-simpler and asserted bit-identical at end-of-training.

**Plasticity active in Phase-3.** The sensory core is frozen (homeostasis θ disabled in both Kok and Richter drivers per Fix A). Only the task-specific task→apical readout splits `W_mh_task_exc` and `W_mh_task_inh` update, via `ThreeFactorRule.delta_mh` (`Δw = lr · memory ⊗ probe_error`, the L23E-space differential modulator introduced in Fix J). Context-memory task weights `W_qm_task` and `W_lm_task` also update (cue binding). Everything else is frozen; `_snapshot_theta(net)` + `_assert_theta_unchanged` are defense-in-depth guards in both Phase-3 drivers.

---

## 3. Key files with brief purpose

```
src/v2_model/
  config.py              ModelConfig, ArchConfig, PlasticityConfig, EnergyConfig, RegimeConfig.
                         Holds all tunable defaults. Vogels targets and homeostasis targets live here.
  layers.py              L23E, L23PV, L23SOM, HE, HPV population classes.
                         FastInhibitoryPopulation (PV/SOM/HPV) stores scalar target_rate_hz, not adaptive θ.
                         BaseExcitatoryPopulation holds ThresholdHomeostasis θ on L23E and HE.
                         All `_make_raw` calls here are the authoritative init_mean registry.
  network.py             V2Network composes all modules. forward() implements sync-Euler with
                         strict pre-update snapshot. plastic_weight_names() is the Phase manifest.
                         frozen_sensory_core_sha() is asserted at checkpoint boundaries.
  context_memory.py      ContextMemory. Task split per v4 (generic + task weights). τ_m=500ms via exp-ODE.
                         W_qm_task, W_lm_task are cue/leader binding; W_mh_task_{exc,inh} are apical readout.
  prediction_head.py     PredictionHead. Linear-combined softplus of (h, c, l23_apical) + b, rectified_softplus.
                         All raw inits at -8.0 post Fix O.
  plasticity.py          UrbanczikSennRule.delta (ε = apical - basal),
                         VogelsISTDPRule.delta (post > ρ → Δraw > 0 under -softplus Dale parameterisation),
                         ThresholdHomeostasis.update (tanh-saturating, deadband=0.2·|ρ|, clamped to ±10),
                         ThreeFactorRule.delta_mh (memory × probe_error outer product, Fix J).
                         All updates return Δraw that will be additively applied.
  energy.py              EnergyPenalty. rate_penalty_delta_drive (L1) and current_weight_shrinkage (L2).
                         current_weight_shrinkage now accepts raw_prior kwarg (Fix P) — defaults to None for legacy behaviour.
  lgn_l4.py              Frozen front-end. DoG + biphasic + Gabor(8 ori) + L4 E/PV divisive norm.
                         Exposes pref_orient_deg helper per-unit (used by Fix K sparse mask).
                         Zero nn.Parameter entries.
  state.py               NetworkStateV2 NamedTuple: r_l4, r_l23, r_pv, r_som, r_h, h_pv, m,
                         pre_traces, post_traces, regime_posterior, ...

scripts/v2/
  train_phase2_predictive.py    Phase-2 driver. PlasticityRuleBank, apply_plasticity_step (one 2-frame window),
                                run_phase2_training (warmup + main loop with rolling-window state + soft_reset).
                                _apply_update respects rule_lr=0 (Fix O) and threads raw_prior for L2 shrinkage (Fix P).
  train_phase3_kok_learning.py  Phase-3 Kok. KokTiming, build_cue_tensor, cue→delay→probe1→blank→probe2
                                (370 steps/trial at dt=5ms), cue_mapping counterbalanced across seeds.
  train_phase3_richter_learning.py  Phase-3 Richter. 6×6 bigram matrix, leader→trailer (500ms+500ms),
                                    deterministic + probabilistic scans. Fix B+C-v2 EMA buffers.
  eval_kok.py                   Kok evaluation. Linear SVM MVPA + mean amplitude + preferred/non-pref asymmetry.
                                Hardened: --fine-offset-deg, --n-localizer-orients, --probe-noise-std.
  eval_richter.py               Richter evaluation. Unit-level amplitude + RSA + preference-rank suppression
                                + pseudo-voxel forward comparison (6-model Richter 2022).
  task74_diagnostics.py         Task #74 probe collection — rate distributions, coverage, SOM check.
  level_1_lgn_l4.py              Level 1: L4 orientation selectivity and pref_orient coverage.
  level_2_l23e_l4_only.py        Level 2: L23E on feedforward drive only (PV/SOM/H/C zeroed).
  level_3_l23e_recurrent.py      Level 3: adds W_rec_l23.
  level_4_l23e_plus_pv.py        Level 4: adds PV loop (divisive).
  level_5_l23e_pv_som.py         Level 5: adds SOM (substrative).
  level_6_full_substrate.py      Level 6: adds H area + context bias.
  level_7_context_memory.py      Level 7: cue-then-delay encoding in m.
  level_8_prediction_head.py     Level 8: pred-head sanity + stream ablation.
  level_9_plasticity_rules.py    Level 9: each plasticity rule in isolation (synthetic inputs vs analytical formula).
  level_10_whole_network_stability.py   Level 10: whole-network with full plasticity, 3000 steps, procedural world.
  _gates_common.py              Shared make_grating_frame, make_blank_frame helpers.

tests/v2/
  [full pytest suite — 641 passed + 1 xfail + 1 warning as of 2026-04-22]
  test_prediction_head_deterministic.py    Pins pred-head b_pred_raw = -8.0 (Fix O).
  test_energy_current_shrinkage_shapes.py  Pins Fix P raw_prior semantics + 6 new tests.
  test_energy_is_global.py                 Pins pathway-agnostic API (forbids pathway/kind/class kwargs).
  test_energy_determinism.py               Pins determinism across calls + seeds.
  test_energy_mask_preservation.py         Pins sparse-mask preservation.
  test_plasticity_vogels.py                Pins post > ρ → Δraw > 0 (authoritative sign convention).

docs/
  task74_handoff.md            This file.
  task73_addendum_som_baseline.md   SOM baseline root-cause (H1 confirmed; pre-Fix-E Phase-2 had r_som ≈ 505).
  v2_model_status.md           Older status digest (pre Task #74 fix chain).
  research_log.md              Chronological project log across sibling branches.
  figures/                     Rendered figures for analyses (Decoder C, HMM, tuning rings, etc.).
  project_summary.md           Higher-level project summary.
```

---

## 4. All fixes applied (chronological)

**Fix A — Phase-3 θ freeze (homeostasis off in task phase).** The Task #37/#45 Debugger showed that L23E homeostasis `target_rate_hz=0.5` and HE `target_rate_hz=0.1` are unreachable from the low-activity operating regime the Phase-3 Kok/Richter trials land in. θ drifted monotonically during Phase-3 trials and dominated the learning signal. Fix A disables `homeostasis.update` on both L23E and HE in the Phase-3 drivers. Implementation: `_snapshot_theta(net)` at trial start and `_assert_theta_unchanged(net, theta_snaps)` at trial end in `scripts/v2/train_phase3_{kok,richter}_learning.py`. HOMEO-OFF isolation collapses the loss-slope divergence 180× (6.9e-5 → 1.2e-6), proving homeostasis is the sole runaway driver during Phase-3. Locked by the `_assert_theta_unchanged` guard in both drivers.

**Fix B — lr scaling (abandoned).** Early attempt to tame Phase-3 divergence by scaling `lr_three_factor` by 0.01×. Insufficient: the underlying problem was homeostasis (Fix A) plus unbounded memory-weight growth (Fix C). The `--lr-task-scale` CLI flag survives in `train_phase3_richter_learning.py` but is not the load-bearing fix.

**Fix C-v1 (additive) and Fix C-v2 (SOM gain) — both abandoned.** Fix C-v1 added a per-(leader × expected) EMA and injected a scalar bias into the L23E apical drive; Fix C-v2 modulated L23 SOM via a learnt gain term. Both were tried before it became clear that SOM was orientation-blind at init (L23SOM has no orientation-biased feedforward mask; its drive is averaged across L23E). SOM modulation therefore could not produce Kok sharpening or Richter dampening signatures because there was no orientation-specific channel to modulate. Do not revisit Fix C-v2.

**Fix D-simpler — freeze SOM excitatory inputs in Phase-2.** The earlier Fix D attempted multi-rule regulation of `W_l23_som_raw` and `W_fb_som_raw` (Vogels + Turrigiano scaling), which fought each other and drove `r_som` to saturation (`r_som ≈ 505` per `docs/task73_addendum_som_baseline.md`). Fix D-simpler freezes both raw tensors at init in Phase-2 (no Vogels, no scaling, no Turrigiano, no rule at all). Enforced in `scripts/v2/train_phase2_predictive.py:apply_plasticity_step` by omitting these weights from the call pattern, and asserted at end-of-training via `torch.equal(net.l23_som.W_l23_som_raw, w_l23_som_init)`.

**Fix E — `W_l23_som_raw` init_mean −1.0 → −4.5.** First pass of SOM stabilisation. Brought `r_som` from `129 Hz` (pre-Fix-E) down to `8.47 Hz`. Evidence in `docs/task73_addendum_som_baseline.md`.

**Fix K — sparse orientation-like-to-like `W_l4_l23` mask + init_mean 4.0 → 1.5.** Level-2 isolation showed that dense uniform `W_l4_l23` saturated every L23E unit at ~30 Hz and eliminated orientation selectivity (`n_pref_bins_5pct = 6/12`, below the ≥8/12 pass criterion; `rate_mean = 30.7` Hz, above the [0.5, 10] band). The fix installs a sparse top-k-per-row retinotopic + orientation-biased mask in `src/v2_model/connectivity.py` and drops `w_l4_l23_init_mean` to 1.5 in `src/v2_model/config.py:83`. Rationale: sparsified drive × smaller init keeps per-unit summed input in biological range (`softplus(1.5) ≈ 1.70` × 15 selected L4 units × mean `r_l4 ≈ 0.2` → ~5 Hz). Level-1 post-Fix-K and Level-2 post-Fix-K both pass.

**Fix L → Fix L2 — L23PV divisive normalisation dose-tuned.** Level-4 isolation (L23E + PV loop) showed `rate_l23e_mean = 0.26 Hz` (below 0.5 floor). Fix L bumped `W_pv_l23_raw` from `-5.0` toward `-5.5`, yielding 0.468 (still below floor). Fix L2 dropped further to `init_mean=-6.5` at `src/v2_model/layers.py:524`. At −6.5 the PV divisive channel is strong enough that the two-population PV↔L23E loop stays under ρ=1 but L23E keeps a firing operating point in-band. Level-4 post-Fix-L2 passes.

**Fix M — `W_l23_som_raw` init_mean −4.5 → −4.2.** After Fix K tightened the L23E rate regime, the post-Fix-E SOM drive was too weak (`rate_som_mean = 0.37`, below the 0.5 floor). Fix M nudges `W_l23_som_raw` up to `-4.2` at `src/v2_model/layers.py:755`. Level-5 post-Fix-M passes.

**Fix N — `W_l23_h_raw` init_mean −5.7 → −5.5.** Level-6 showed `rate_h = 0.046` (below the 0.05 floor) once the full substrate was active with the post-Fix-K L23E regime. Fix N bumps `W_l23_h_raw` to `-5.5` at `src/v2_model/layers.py:861`. Level-6 post-Fix-N passes.

**Fix O — pred-head inits to clamp bounds + `lr=0` is true no-op.** Debugger claimed that pred-head weights at `init_mean=-10` fell outside the `[-8, 8]` raw clamp in `apply_plasticity_step` and got snapped to −8 on step 1 (a 7.4× softplus gain jump). Part 1 of Fix O changed all four pred-head inits from −10.0 → −8.0 at `src/v2_model/prediction_head.py:217/224/229/234`. Part 2 added a `rule_lr` kwarg to `_apply_update` in `scripts/v2/train_phase2_predictive.py` — when `rule_lr == 0.0`, the function is a complete no-op (no weight decay, no energy shrinkage, no clamp). This makes single-rule ablation (`lr_urbanczik=0`, etc.) bit-identical to skipping that rule entirely. Validation: `logs/level10_post_fixO.json` showed rate trajectories bit-identical to pre-Fix-O, which *falsified* the Debugger's root-cause hypothesis about the pred-head path driving L23E collapse (x̂ does not feed back into L23E dynamics — it only enters the pred-head's own U-S signal). Fix O is retained as a legitimate cleanliness improvement even though it did not resolve Level 10. Locked by `test_bias_init_deterministic_across_seeds` and the new `rule_lr=0` guard path.

**Fix P — `EnergyPenalty.current_weight_shrinkage` anchored at `raw_prior`.** Previously shrinkage was `Δw = −w · s / (1+s)` with `s = β · mean_b(pre²)`, pulling raw weights toward zero. For strongly-negative inits (`W_pv_l23_raw=−6.5`, pred-head weights `−8`, Dale inhibitory `−5`), shrinkage toward zero *grew* the effective softplus output of those weights — the opposite of what "regularise deviation from rest" should do. Fix P adds a `raw_prior: Optional[Tensor] = None` kwarg; when provided, `Δw = −(w − raw_prior) · s / (1+s)`, inert at the init. Callers in `_apply_update` thread `raw_prior = _raw_prior(net, module, weight, w)` through automatically (matching the Task #50 weight-decay anchor pattern). Level-10 post-Fix-P shows L23PV fully in range (24 → 38 Hz), HE/HPV much improved, orientation 11/12 bins, collapse kinetics qualitative change from abrupt-by-step-300 → gradual monotone drift. Pytest 641 passed (+6 new Fix P unit tests). Locked by `test_raw_prior_at_init_is_inert`, `test_raw_prior_pulls_toward_prior_not_zero`, `test_raw_prior_none_matches_legacy_shrink_to_zero`, and three error-path tests.

**Fix Q' — freeze `h_pv.W_pre_raw` by analogy with Fix Q.** HE→HPV is the symmetric E→I excitatory synapse in the H area; the same Vogels-sign-inversion structural bug applies even though HPV rates were empirically stable pre-Q' (the anti-homeostatic pressure is latent). Implemented via `freeze_W_pre=True` on `self.h_pv` in `network.py`; the h_pv Vogels entry in `apply_plasticity_step` is removed (the H-inhibitory for-loop now only contains `h_e.W_pv_h_raw`). Manifest test updated; the Fix Q pytest lock file is extended with `test_h_pv_W_pre_also_frozen_fix_q_prime` and `test_phase2_training_leaves_h_pv_W_pre_unchanged`. Same dispatch also relaxed Level-10 rate-range floors to match the empirical substrate equilibrium (L23E floor 0.3 → 0.2 Hz; L23SOM floor 0.3 → 0.05 Hz; other gates unchanged). Level 10 **verdict=PASS** post-Fix-Q' with `l23e_final=0.309`, `l23e_cv_last1k=0.038`, `som_final=0.125`, `pv_final=23.622`, `he_final=0.035`, `hpv_final=5.223`, `theta_l23e_drift=0.002`, `n_preferred_bins_final=12/12`, `loss_trend=flat`, `issue_if_fail=none`. Pytest 645 passed (+1 new test). Substrate validated end-to-end under Phase-2 plasticity for 3000 steps.

**Fix Q — freeze `l23_pv.W_pre_raw` in Phase-2 (wrong-rule-on-wrong-synapse).** The Debugger isolated `rules.vogels_ipop` on `l23_pv.W_pre_raw` as the Level-10 collapse driver. That synapse is L23E→L23PV — an *E→I excitatory* weight. Vogels iSTDP is designed for I→E: the homeostatic sign is baked into the caller's `-softplus` Dale wrapper, so on an E→I wrapped as `+softplus` the term inverts, drives L23E monotone collapse and SOM exponential silencing. Cleanest intervention — matching Fix D-simpler for L23SOM excitatory inputs — is to freeze at init end-to-end. Implemented via a `freeze_W_pre: bool = False` kwarg on `FastInhibitoryPopulation` (passed `True` only for `self.l23_pv`) so `plastic_weight_names()` excludes `W_pre_raw`; the Vogels loop entry in `apply_plasticity_step` is removed. `h_pv.W_pre_raw` is out of Fix Q scope (same structural concern but not isolated). Locked by `tests/v2/test_l23_pv_W_pre_frozen_phase2.py`. Pytest 644 passed. Level-10 post-Fix-Q: L23E CV_last1k=0.038 (stable ~0.30 Hz vs. pre-Q monotone 0.328→0.097), SOM stable ~0.12 Hz (vs. silenced to ~0), PV/HE/HPV all in range, loss_trend flat (vs. up), 12/12 bins. Verdict FAIL only on rate-range thresholds (L23E 1/10 just under 0.3; L23SOM 0.12 < 0.3 floor). Pathology eliminated; remaining deltas are range-calibration.

---

## 5. Validation state — Level 1 through 10

The bottom-up validation protocol introduced by Lead's dispatch in Task #74 isolates one component at a time by zeroing every other input stream and freezing plasticity, then asks a targeted question about that component's behaviour. Each level has its own script in `scripts/v2/level_N_*.py`, writes a JSON verdict to `logs/task74/level_N*.json`, and reports a DM line of the form `levelN_verdict=<pass|fail|neutral_baseline> …`. The current verdicts are:

```
Level 1  LGN/L4 orientation coverage                         PASS (post-Fix-K: logs/task74/level_1_post_fixK.json)
Level 2  L23E on feedforward drive only                       PASS (post-Fix-K: logs/task74/level_2_post_fixK.json)
Level 3  + W_rec_l23 recurrent loop                           PASS (as-is:     logs/task74/level_3_l23e_recurrent.json)
Level 4  + L23PV loop (divisive normalisation)                PASS (post-Fix-L2: logs/task74/level_4_post_fixL2.json)
Level 5  + L23SOM (substrative, context-agnostic at init)     PASS (post-Fix-M: logs/task74/level_5_post_fixM.json)
Level 6  + HE/HPV + context bias                              PASS (post-Fix-N: logs/task74/level_6_post_fixN.json)
Level 7  ContextMemory cue-then-delay encoding                PASS (neutral-baseline branch allowed per spec — decoder acc ≈ 0.5 at init is
                                                                  expected because W_mh_task_{exc,inh}=0, so the readout pathway is silent;
                                                                  Phase-3 training installs the mapping. logs/task74/level_7.json)
Level 8  PredictionHead sanity + stream ablation              PASS as neutral_baseline (logs/level8_prediction_head.json)
                                                                  — all three streams demonstrably engage (h=2.62e-2, c=1.37e-3, apical=9.61e-1
                                                                  on Δx̂/‖x̂‖); ‖x̂‖ > 0 and ≪ 5× copy_last norm; quantitative gates don't
                                                                  hit "pass" at init because x̂ magnitudes are correctly tiny (softplus(-8)
                                                                  pre-training); Lead chose "Option A: accept as neutral_baseline".
Level 9  Plasticity rules in isolation                        PASS on all 4 rules (logs/level9_plasticity_rules.json). Urbanczik-Senn: 16×16
                                                                  cells, 100% sign match, machine-precision magnitude. Vogels: 16×8 cells, sign
                                                                  matches code convention (post > ρ → Δraw > 0 under -softplus Dale). Homeostasis:
                                                                  tanh-saturating Δθ = lr·tanh(err/scale)·scale with deadband 0.2·|ρ|, matches
                                                                  analytical formula. Fix J: delta_mh matches l23e_modulator × m outer product,
                                                                  spot-checks pass at clamped ±0.01 boundaries.
Level 10 Whole-network stability (3000 plasticity steps)       PASS (post-Fix-Q'+gate-relax: logs/level10_post_fixQprime.json)
```

A "pass" means: the DM line ends with `issue_if_fail=none` (or `issue_if_fail=none (neutral_baseline reason)` for accepted-null branches), the JSON has `"verdict": "pass"` or `"verdict": "neutral_baseline"`, and all gated thresholds meet Lead's per-level criteria.

---

## 6. Level 10 — substrate validated (was open failure)

Level 10 runs 3000 Phase-2 training steps with all plasticity rules active on a procedural-world batch of 4 trajectories. It passes if every population's mean rate stays in biological range for the full run, CV over the last ~1000 steps < 2, θ drift < 0.5, orientation selectivity ≥ 8/12 preferred-orientation bins maintained, and loss trend is DOWN or FLAT (not UP). The final state after Fixes Q + Q' and Lead's gate-floor relaxation (2026-04-22):

```
verdict              PASS
l23e_rate_final      0.309 Hz    (range [0.2, 5.0]   — stable, CV_last1k = 0.038)
l23pv_rate_final    23.622 Hz    (range [5.0, 60.0]  — stable)
l23som_rate_final    0.125 Hz    (range [0.05, 5.0]  — stable, no silencing)
he_rate_final        0.035 Hz    (range [0.02, 1.0]  — stable)
hpv_rate_final       5.223 Hz    (range [3.0, 50.0]  — stable)
theta_l23e_drift     0.002       (PASS < 0.5)
n_preferred_bins     12/12       (PASS ≥ 8/12 — full orientation coverage)
fwhm_trajectory      same        (selectivity preserved)
loss_trend           FLAT        (no regression; Phase-2 training is neutral, not divergent)
issue_if_fail        none
```

Root cause of the prior failure was confirmed by the Debugger: `rules.vogels_ipop` applied to `l23_pv.W_pre_raw` (L23E→L23PV, an E→I excitatory synapse) — Vogels iSTDP is designed for I→E and the sign convention is baked into the caller's `-softplus` Dale wrapper, so on an E→I wrapped as `+softplus` the homeostatic term inverts and becomes anti-homeostatic, driving L23E monotone collapse and SOM exponential silencing. Fix Q froze `l23_pv.W_pre_raw` at init; Fix Q' extended the same freeze to `h_pv.W_pre_raw` (HE→HPV, same structural bug, latent pressure). Lead's gate-floor relaxation (L23E 0.3→0.2, L23SOM 0.3→0.05) aligns the gate with the empirical substrate equilibrium. Output JSON at `logs/level10_post_fixQprime.json`.

---

## 7. Full history of what was tried that didn't work

Keep this section — it is the single best protection against an incoming agent re-trying paths that are already known dead ends.

Task #70 (W_qm / W_mh init bootstrap) addressed a finding that the task weights `W_mh_task_{exc,inh}=0` at init meant the cue pathway was silent, and proposed a small-random bootstrap so Phase-3 had a starting gradient. Implemented and tested. Phase-3 Kok showed a weak Δpeak signal but the sensitivity gates (eval_kok with its original tolerances) reported "null" because the fine-offset condition was unchanged. See `logs/task74/baseline_phase3_kok_s42.*` and `checkpoints/v2/phase3_kok_task70/`.

Task #72 upgraded the evaluation harnesses (`eval_kok.py` hardened with `--fine-offset-deg`, `--n-localizer-orients`, `--probe-noise-std`; `eval_richter.py` with preference-rank binning and 6-model pseudo-voxel forward comparison). The upgraded eval produced a *still-null* result on Task #70's checkpoint. See `logs/task74/eval_kok_task74.*`, `logs/task74/eval_richter_D.log`, `checkpoints/v2/phase3_kok_task74/`.

Task #73 launched a debug cascade and found three stacked root causes, labelled H_coverage, H_magnitude, H_direction. H_coverage: L23E preferred-orient histogram had bins with zero units so some Kok cue orientations had no preferred readout. H_magnitude: `W_mh_task_*` amplitudes were too small to produce detectable modulation (the Task #70 bootstrap init was too conservative). H_direction: the SOM pathway was being modulated but SOM has no orientation-biased drive, so the modulation couldn't produce the Kok or Richter signature. See `docs/task73_addendum_som_baseline.md` and `scripts/v2/_debug_task73_kok15.py`, `scripts/v2/_debug_task73_dx58.py`, `scripts/v2/_debug_task73_dx6.py`.

Task #74 Fix C-v1 attempted an additive per-(leader × expected) bias injected into L23E apical drive. Abandoned because it didn't address H_direction — the modulation still went through SOM without orientation structure. Task #74 Fix C-v2 tried to install a learnable SOM gain, likewise abandoned for the same reason. Task #74 Fix J replaced SOM modulation with a differential modulator in L23E-space (`ThreeFactorRule.delta_mh` with `l23e_modulator × m` outer product) which addressed H_direction. Fix J is the current Phase-3 readout mechanism.

SOM baseline debug (documented in `docs/task73_addendum_som_baseline.md`). Pre-Fix-E Phase-2 showed `r_som ≈ 505 Hz` (max 3176) with `W_l23_som_raw` drifting from −1.0 → +8.0 during training. Four hypotheses tested: H1 fan-in miscalibration (CONFIRMED, 25.6× inflation), H2 θ_som adaptation (FALSIFIED — L23SOM is FastInhibitoryPopulation which has no adaptive θ), H3 softplus linear regime (CONFIRMED — 475× threshold ratio), H4 divisive stabiliser (CONFIRMED MISSING). Fix E (`W_l23_som_raw` init_mean −1.0 → −4.5, then Fix M to −4.2) resolved H1+H3. Fix D-simpler (freeze SOM excitatory inputs in Phase-2) addressed the architectural enabler H4 by preventing Vogels/Turrigiano from re-inflating the weights.

Fix H-global (single scalar Vogels target of 3.0 for all Vogels rules) broke HE homeostasis — HE has `target_rate_hz=0.1` and a Vogels target of 3.0 drove HE toward 3.0 while homeostasis lowered θ_h unboundedly (Debugger #37). Resolved by splitting Vogels targets per post-population: `vogels_target_l23e_hz=3.0`, `vogels_target_h_hz=0.1`, `target_rate_hz=1.0` (I-pop self-regulation).

Fix-B lr scaling at 0.01× was insufficient because it masked but did not resolve the underlying Phase-3 divergence. The real drivers were Fix A (homeostasis freeze) + Fix C-v2/J (direction).

The `xfail` in `tests/v2/test_predictive_loss_slope.py::test_predictive_loss_slope_is_negative_over_1000_steps` is a known-open issue: the conservative stability-first initialization does not produce detectable Phase-2 loss decrease in 1000 steps. A passing Level 10 should close this xfail.

---

## 8. Plan forward

Immediate, next one to three Lead dispatches. Resume the Vogels per-weight ablation paused for this handoff. The dispatch should be: (a) run Level 10 with `lr_vogels_l23=0` while keeping all other rules at `1e-4`, then `lr_vogels_h=0`, then `lr_vogels_ipop=0`; (b) for whichever ablation prevents the L23E monotone drift, identify that rule as the driver; (c) for that rule, print Δraw at step 1 on a representative weight and check whether the sign is consistent with `post < ρ → Δraw < 0` (i.e. Vogels lowering inhibition when E is under target). Once the driver is confirmed, the fix is likely a parameter tweak (target rate, lr, or weight-decay anchor) rather than an architectural change, because Levels 1–9 already lock the component behaviour in isolation. Re-run Level 10 with the targeted fix; expected result is all populations in range, orientation ≥ 8/12, loss trend flat or down. If Level 10 passes, the substrate is fully validated end-to-end with plasticity and we move to Phase-3.

Then, in order. First, run the full Phase-2 training for 3000 steps on the validated substrate and produce a clean post-Phase-2 checkpoint in `checkpoints/v2/phase2_post_fixes/`. This replaces the older `checkpoints/v2/step1_fixE_init.pt` as the canonical starting point. Second, run Phase-3 Kok with the Fix J differential modulator, at 0.1× baseline lr (`lr_three_factor_kok=1e-4`), for the full trial count. The canonical Kok checkpoint becomes the starting point for all ex/unex evaluations. Third, run `eval_kok.py` with the hardened flags (`--fine-offset-deg 15`, `--n-localizer-orients 36`, `--probe-noise-std 0.1`) so the tolerance is tight enough to detect the 15° offset as "fine". Fourth, compute the expected-vs-unexpected triad per `memory/feedback_expectation_metric_triad.md`: decoding accuracy, net activity, and response shape (peak at true-ch, FWHM, preferred-rank asymmetry) on expected vs unexpected probes. Fifth, if Kok shows a robust dampening or sharpening signature, mirror the protocol to Richter: Phase-3 Richter training + `eval_richter.py` hardened + triad metrics. Sixth, if Kok is null, do not declare a null result — return to the bottom-up debug cascade before attempting another Phase-3 run, per `memory/feedback_never_propose_null.md`.

Long-term, the project publishes a circuit-grounded adjudication between Kok 2012 sharpening and Richter 2018/2019 dampening. The deliverable is a single substrate that produces one signature robustly across seeds and fails to produce the other under the same circuit parameters, with a clear mechanistic explanation tied to the three-factor rule's gain polarity on `W_mh_task_exc` vs `W_mh_task_inh`.

---

## 9. Rules and user preferences

The user expects a strict bottom-up systematic approach and a strict agent-team division of labour. The Lead (main thread) orchestrates, delegates, and reports — the Lead never reads files, runs tests, edits code, or debugs directly. The Researcher explores and produces plans without editing files. The Debugger confirms root causes with evidence-only protocol — no opinions, no guessing, no blaming without tested isolating experiments. The Coder implements minimal patches against a confirmed root cause, never a guess. The Validator issues GO/NO-GO verdicts with hard evidence and does not edit code or tests. Every mismatch, failure, or unexpected result must be handed to the Debugger; the Lead never debugs on its own. These rules are documented in the two `CLAUDE.md` files referenced at session start.

User-specific preferences retrieved from memory. `memory/feedback_never_propose_null.md` is absolute: never surface "accept the null result" as a path forward; always dispatch the next debug step instead. `memory/feedback_prose_not_bullets.md` is also absolute: always explain in concise prose paragraphs, not bulleted lists of fragments (this handoff doc obeys that rule; the listings here are references, tables, or code, not fragments of explanation). `memory/feedback_expectation_metric_triad.md` requires the decoding + activity + shape triad for every expected-vs-unexpected claim. `memory/feedback_scientific_rigor.md` enumerates seven rules for plan review: arithmetic fidelity, scope discipline, no pre-commit on ablations, model-vs-claim validity split, no training-task leakage, avoid circular benchmarks, and match paradigm structure exactly.

Additional agent-memory rules live in `~/.claude/agent-memory/coder/`. `feedback_confirm_before_long_runs.md` requires a sized-up proposal to Lead (what/why/estimated time/gate/commit hash) and approval before any job over ~5 minutes of wall time. `feedback_execute_dispatch_literal.md` requires running Lead's fenced bash blocks verbatim. `feedback_halt_freeze.md` requires stopping immediately on HALT — even if the fix seems obvious, the Debugger must verify state before any patch lands. `feedback_atomic_edits.md` requires landing all edits of a cross-file invariant (module math + its test's closed-form) before declaring done. `feedback_flag_cross_scope_items.md` requires flagging any brief item that crosses a deferred scope line rather than silently including or omitting it.

Summary-packed DM convention. The `summary` field of every `SendMessage` must carry the load-bearing metadata (verdict, metric values, file paths, commit hash) because the message body does not always land intact. The body carries prose context and full numeric tables; the summary is what the Lead sees at a glance even under truncation.

---

## 10. Artifacts and log locations

Checkpoints live under `checkpoints/v2/`. The `step1_fixE_init.pt` file is the clean Fix-E init used by Phase-3 evaluations before Task #70's W_qm/W_mh bootstrap was applied; it is the canonical "pre-training" starting point for sanity probes. `phase3_kok_fixJ_init_s42/` and `phase3_kok_fixJ_mod_s42/` hold Phase-3 checkpoints with Fix J; `phase3_kok_task68/`, `phase3_kok_task70/`, `phase3_kok_task74/`, `phase3_kok_task74B/`, `phase3_kok_task74C/`, `phase3_kok_task74D/` are historical Kok runs at increasing fix maturity. `phase3_richter_task68/`, `phase3_richter_task70/`, `phase3_richter_task74D/` are the corresponding Richter runs. The latest canonical checkpoint for production evaluation is `phase3_kok_task74D/` (Fix J + Fix A + W_mh bootstrap + B/C-v2 EMA buffers).

Logs live under `logs/task74/`. Level JSON outputs are `level_N.json` (pre-fix), `level_N_post_fix{K,L,L2,M,N}.json` (after each targeted fix), and the in-progress Level 10 results are `logs/level10_whole_network_stability.json` (pre-Fix-O/P, the original failing run), `logs/level10_post_fixO.json` (after Fix O, bit-identical rate trajectory to pre-Fix-O — falsifying the pred-head hypothesis), `logs/level10_post_fixP.json` (after Fix P, partial win described in §6). Ablation diagnostics under `logs/task74/level10_{ablation,clean_ablation,diff_probe,probe_driver,vogels_per_weight,vogels_subrule}.{json,log}` — these are the in-flight Vogels ablation artefacts that the dispatch paused. Phase-3 evaluation outputs are `logs/task74/eval_kok_{task74,task74D}.*` and `logs/task74/eval_richter_{D,task74D}.*`.

Documents live under `docs/`. `task73_addendum_som_baseline.md` is the SOM root-cause evidence document referenced throughout this handoff. `research_log.md` is the chronological project log (older entries are from sibling dampening-analysis and failed-dual-regime-experiments branches). `v2_model_status.md` is an older status digest that predates the Task #74 fix chain and should be read with that caveat — it reflects pre-fix state. `project_summary.md` and `rescues_1_to_4_summary.md` are higher-level overviews.

---

## 11. Config and numeric state (as of 2026-04-22)

**Per-weight init_mean registry.** L23E.W_rec_raw=−5.0; L23E.W_pv_l23_raw=−6.5 (Fix L2); L23E.W_som_l23_raw=−5.0; L23E.W_fb_apical_raw=−5.0; L23E.W_l4_l23_raw=1.5 (Fix K, with sparse top-k mask). L23SOM.W_l23_som_raw=−4.2 (Fix M, frozen in Phase-2 per Fix D-simpler); L23SOM.W_fb_som_raw=−5.0 (frozen in Phase-2). HE.W_l23_h_raw=−5.5 (Fix N); HE.W_rec_raw=−5.0; HE.W_pv_h_raw=−5.0. L23PV.W_pre_raw has `w_pre_init_mean=-1.0` set in network.py:287; HPV.W_pre_raw has `w_pre_init_mean=3.0` set in network.py:337. PredictionHead.{W_pred_H_raw, W_pred_C_raw, W_pred_apical_raw, b_pred_raw} all = −8.0 (Fix O). Context-memory generic weights (W_hm_gen, W_mm_gen, W_mh_gen) init `normal(mean=0, std=0.1)`. Task-split weights (W_qm_task, W_lm_task, W_mh_task_exc, W_mh_task_inh) currently at init per Task #70 bootstrap (`normal(0, 0.3)` per task_input_init_std); consult `src/v2_model/context_memory.py`.

**Vogels iSTDP targets.** `vogels_target_l23e_hz = 3.0` for L23 I→E synapses. `vogels_target_h_hz = 0.1` for H I→E synapses. `target_rate_hz = 1.0` for I-pop self-regulation (l23_pv.W_pre_raw, h_pv.W_pre_raw) and for the FastInhibitoryPopulation (PV/SOM/HPV) ReLU threshold. Defined in `src/v2_model/config.py:119,131,132`.

**Homeostasis targets (per adaptive-θ excitatory pop).** L23E `target_rate=0.5` (set in network.py:253). HE `target_rate=0.1` (set in network.py:318). `lr_homeostasis=1e-5`. ThresholdHomeostasis applies `Δθ = lr · tanh(err/scale) · scale` with `scale = 0.1·|ρ| + 1e-3` and a deadband of `0.2·|ρ|`; θ is clamped to `±10`. Under Phase-3, `homeostasis.update` is not called (Fix A); `_assert_theta_unchanged` guards.

**Plasticity learning rates.** `lr_urbanczik_senn = 1e-4`; `lr_vogels_istdp = 1e-4`; `lr_homeostasis = 1e-5`; `lr_three_factor_kok = 1e-3`; `lr_three_factor_richter = 1e-3`. `weight_decay = 1e-5`. Per-step Δw clamp is `±0.01` inside the rule's `.delta()` method. Raw-weight clamp is `[-8, 8]` applied in `_apply_update` (and respected as a no-op when rule_lr==0 per Fix O).

**Energy penalty.** `alpha_rate = 1e-3` (L1 subtractive contribution to firing E unit drives). `beta_syn = 1e-4` (L2 implicit-Euler shrinkage factor scaling `β·mean_b(pre²)`). Shrinkage now anchored at raw_prior per Fix P.

**Phase-2 rolling-window policy.** `warmup_steps = 30` (forward-only, no plasticity, to let activity propagate). `segment_length = 50` (soft-reset interval). `soft_reset_scale = 0.1` (multiplies rate tensors, pre_traces, post_traces; regime_posterior reset to uniform). `batch_size = 4` with per-element persistent worlds and per-element `WorldState`.

**Procedural-world seeds.** `SEED_BASE_TRAIN = 42` drives the "train" seed_family; `SEED_BASE_EVAL = 9000` drives the "eval" family (so `trajectory_seed=1` under eval gives world RNG seed 9001). Regime-switch probability 0.02; low-hazard jump 0.05; high-hazard jump 0.30; drift step 5°. `n_regimes = 4` (CW-drift, CCW-drift, low-hazard, high-hazard).

**Population sizes.** L4 E=128, L4 PV=16, L23 E=256, L23 PV=16, L23 SOM=32, HE=64, HPV=8, context memory `n_c = n_m = 48`, prediction-head output n_l4_e=128. Retinotopy 4×4 = 16 pool cells × 8 orientations = 128 L4 E.

---

## 12. How to run things

Full pytest suite:

```
PYTHONPATH=. python3 -m pytest tests/v2/ -q
```

Expect 641 passed, 1 xfailed, 1 warning in ~4 minutes on WSL Linux. The single xfail is `test_predictive_loss_slope_is_negative_over_1000_steps` and is documented in §7.

Re-run a single validation level (replace `N` with 1..10):

```
PYTHONPATH=. python3 scripts/v2/level_N_<name>.py --output logs/task74/level_N_<tag>.json
```

For example:

```
PYTHONPATH=. python3 scripts/v2/level_1_lgn_l4.py --output logs/task74/level_1_post_fixK.json
PYTHONPATH=. python3 scripts/v2/level_10_whole_network_stability.py --output logs/level10_post_fixP.json
```

Phase-2 training 3000 steps (canonical command):

```
PYTHONPATH=. python3 -m scripts.v2.train_phase2_predictive \
  --seed 42 --n-steps 3000 --batch-size 4 \
  --lr-urbanczik 1e-4 --lr-vogels 1e-4 --lr-hebb 1e-4 \
  --weight-decay 1e-5 --beta-syn 1e-4 \
  --warmup-steps 30 --segment-length 50 --soft-reset-scale 0.1 \
  --out-dir checkpoints/v2/phase2
```

Outputs: `checkpoints/v2/phase2/phase2_s42/metrics.jsonl` and `step_N.pt` at each 100-step checkpoint.

Phase-3 Kok training:

```
PYTHONPATH=. python3 -m scripts.v2.train_phase3_kok_learning \
  --init-checkpoint checkpoints/v2/phase2/phase2_s42/step_3000.pt \
  --seed 42 --n-trials 400 \
  --lr-three-factor 1e-4 \
  --out-dir checkpoints/v2/phase3_kok_<tag>
```

Phase-3 Kok evaluation (hardened):

```
PYTHONPATH=. python3 -m scripts.v2.eval_kok \
  --ckpt checkpoints/v2/phase3_kok_<tag>/final.pt \
  --seed 42 --n-trials 100 \
  --fine-offset-deg 15 --n-localizer-orients 36 --probe-noise-std 0.1 \
  --output logs/task74/eval_kok_<tag>.json
```

Phase-3 Richter training:

```
PYTHONPATH=. python3 -m scripts.v2.train_phase3_richter_learning \
  --init-checkpoint checkpoints/v2/phase2/phase2_s42/step_3000.pt \
  --seed 42 --n-trials 400 \
  --lr-three-factor 1e-4 \
  --out-dir checkpoints/v2/phase3_richter_<tag>
```

Phase-3 Richter evaluation:

```
PYTHONPATH=. python3 -m scripts.v2.eval_richter \
  --ckpt checkpoints/v2/phase3_richter_<tag>/final.pt \
  --seed 42 --output logs/task74/eval_richter_<tag>.json
```

Task #72-style sanity evaluation on an arbitrary checkpoint:

```
PYTHONPATH=. python3 -m scripts.v2.task74_diagnostics \
  --ckpt <path> --seed 42 \
  --output logs/task74/diag_<tag>.json
```

Vogels per-weight ablation (paused diagnostic; rerun each command separately):

```
PYTHONPATH=. python3 scripts/v2/_debug_task74_vogels_subrule.py \
  --disable-vogels-l23 --output logs/task74/level10_ablate_vogels_l23.json

PYTHONPATH=. python3 scripts/v2/_debug_task74_vogels_subrule.py \
  --disable-vogels-h --output logs/task74/level10_ablate_vogels_h.json

PYTHONPATH=. python3 scripts/v2/_debug_task74_vogels_subrule.py \
  --disable-vogels-ipop --output logs/task74/level10_ablate_vogels_ipop.json
```

(Consult the script for the exact CLI flags; it lives at `scripts/v2/_debug_task74_vogels_subrule.py` and was the in-flight artefact at the moment of this handoff.)

---

End of handoff. The incoming agent should be able to read this document, open `scripts/v2/level_10_whole_network_stability.py`, `scripts/v2/train_phase2_predictive.py:apply_plasticity_step`, and `src/v2_model/plasticity.py`, and resume the Vogels per-weight ablation within 30 minutes. The current pytest state is green (641 passed). The current blocker is Level 10 L23E monotone drift + SOM exponential silencing, partially remediated by Fix O + Fix P, with the remaining driver to be identified by clean single-rule ablation now that `lr=0` is an honest no-op.
