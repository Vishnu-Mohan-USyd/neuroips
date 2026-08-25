# RUN_LOG — fromscratch_joint_20260825 (coder)

- 2026-08-25T03:18:22Z SETUP. Protocol: /home/vishnu/neuroips_analysis/fromscratch_joint_20260825/PROTOCOL.md.
  Source harness: /home/vishnu/scratch/flank_sharpening_20260819/harness/ (produced the ladder_s0p04 runs; launch pattern per phase5_run_seed_pair.sh).
  simple_net.py sha 511581a640526a9bdbfca9effc72f60420211ee3825d7449162667e81e716f74 (byte-copy, = frozen lib).
  tuned_emergence_lib.py sha 3024bf0718ba69231e60f6a807cde2bfda0e10218519f6c5b7319ae222110e7a (byte-copy, = frozen lib).
  train_sweep.py on disk today is the 'official' state (sha 9db8f975..., pred_inhib 0.05); the s0p04 ladder used the transient s0p04 state (0.04/4.0).
  Reconstruction: official copy + single sed 0.05->0.04 => harness/train_sweep_s0p04_source.py, sha VERIFIED equal to the recorded s0p04-state sha 7eb46f6c2a3b22885574b3961ce97ba9a1224259dc6654075cc8421b0e25d821. Donor checkpoint tuned_net_config confirms 0.04/4.0, rnn_tanh, task_weight 0.5, freeze_local_comp True, posterior_prior_excess.

## Diff audit — harness/train_sweep.py (run file) vs harness/train_sweep_s0p04_source.py (byte-exact s0p04 source)
```
729a730,735
>             # FROMSCRATCH DELTA (d): numbered weight snapshots every 500
>             # steps for trajectory analysis.
>             if step % 500 == 0:
>                 atomic_torch_save(
>                     payload, run_dir / f"alpha_{slug}_step{step:05d}.pt"
>                 )
908,909c914,928
<         common_state, references = run_pretrain(
<             args, run_dir, device, event_log
---
>         # FROMSCRATCH DELTA (a): skip the pretrain stage entirely — the joint
>         # arm starts from the fresh seed init. Same lineage as run_pretrain's
>         # opening (seed_everything -> build -> reference_values), so the init
>         # state and references are bitwise those of the two-stage run.
>         seed_everything(args.seed)
>         fresh_net = tuned.build_tuned_from_config(MODEL_CONFIG).to(device)
>         references = reference_values(fresh_net, device)
>         common_state = copy.deepcopy(fresh_net.state_dict())
>         del fresh_net
>         event_log.write(
>             {
>                 "event": "fromscratch_init",
>                 "state_sha256": state_sha256(common_state),
>                 "references": references,
>             }
```

  Delta mapping: (a) = the fromscratch_init hunk replacing the run_pretrain call (init lineage preserved: seed_everything -> build -> reference_values, bitwise the two-stage init). (b) = NO source change needed: run_alpha already trains the FULL arm policy set_axis_parameter_policy = gru (cell) + W_fb + circ_raw, local_comp frozen (freeze_local_comp default True) — the pretrain policy never executes. (c) = CLI flag --axis-steps 11000 (budget-matched 3000+8000). (d) = the numbered-snapshot hunk (every 500 steps; 22 snapshots + final). Nothing else differs.

- 2026-08-25T03:18:42Z LAUNCH: ./run_joint.sh — python3 -B harness/train_sweep.py --out runs/joint --seed 8 --alphas 0.5 --axis-steps 11000 --recurrent-cell rnn_tanh --device cuda:0 (PYTHONHASHSEED=0; mem gate + 8GB RSS watch as in the source runner; all other args = source defaults: batch 128, seq 12, lr 1e-3, clip 5.0, task_weight None -> 0.5, freeze_local_comp True, feedback posterior_prior_excess).
- 2026-08-25T03:19:19Z START OK: run_start config exact (0.04/4.0, rnn_tanh, tw 0.5, freeze_local_comp, posterior_prior_excess); fromscratch_init sha 08bbb3ae58ac2bebfd46591f0934154af7e6b8fea30663c5ce5259564f5fdf5a; init k +0.5457 == the recorded pretrain k (circ_raw untouched in pretrain => init k IS pretrain k — lineage cross-check); references bitwise == donor checkpoint references (fresh-init lineage PROVEN). Training under way.

- 2026-08-25T03:25:00Z RUN COMPLETE (exit 0, peak RSS 2.3 GB): 22 numbered snapshots + final + training.jsonl under runs/joint/seed_8/. MEASURED (scripts/measure_joint.py; transplant-study pipeline, comparators from n4_assay.json seed8 alpha0.5 PPP/TTT; init reconstruction sha-verified == run's logged init sha; init references bitwise == donor references): results_joint.json.
  ENDPOINT (joint | two-stage TTT | host PPP): M 0.2487|0.2961|1.3554, center 0.1411|0.1436|1.8070, flank 0.3879|0.4999|0.8102, H 0.1759|0.1991|0.4769, decode_A_minus_B -0.2018|-0.0285|+0.7578, decode_A 0.1766|0.2859|0.7603, rate 0.0415|0.0493|0.2259, k -3.2479|-3.5016|+0.5457. vitality PASS, CE max 2.49 (untripped).
  ORDER: k first negative at eventlog step 700 (600:+0.29 -> 700:-0.22); decode_A_minus_B NEVER task-informative (max +0.0007 @500, then negative, final -0.2018); decode_A falls 0.710(init)->0.177(final). Rates 0.188->0.167(500)->0.027(1000)->0.0415(final). Curves: task 0.504->0.371, energy 1.119->0.327, next_ce 3.594->2.189.
  NOTE: rate_A/rate_B fields in trajectory rows are null (assay summary lacks those key names; continuation_mean_rate + B_minus_A_rate carry the rate readouts).

- 2026-08-25T04:47:22Z FOLLOW-UP LAUNCH (lead): identical variant at alpha 0.0 — ./run_joint_a0p0.sh = run_joint.sh with only --alphas 0.0, --out runs/joint_alpha0p0, log/kill filenames (diff above one-for-one). Same harness file, same deltas, same envelope.

- 2026-08-25T04:51:56Z ALPHA0.0 FOLLOW-UP COMPLETE (exit 0, peak RSS 2.3 GB; init sha == 08bbb3ae... same as a0.5 run — init is pre-alpha). Measured (measure_joint.py alpha0.0 -> results_joint_alpha0p0.json). ENDPOINT (joint | two-stage TTT | host): M 0.8813|0.9672|1.3554, center 0.9020|1.1895|1.8070, flank 0.8549|0.8279|0.8102, H 1.0000|0.9907|0.4769, decode_A_minus_B +0.0475|+0.3825|+0.7578, rate 0.1469|0.2259(host), k -0.0084|+0.0473|+0.5457. vitality PASS, CE untripped. k: +0.5457 -> +0.0289@500 -> first negative @~900 -> hovers -0.008..-0.019 (settles ~0, OFF). decode: rises only to ~+0.05..0.09 mid-run, final +0.0475 (vs two-stage 0.3825). Sharpening phenotype (center boost) ABSENT despite perfect placement.
