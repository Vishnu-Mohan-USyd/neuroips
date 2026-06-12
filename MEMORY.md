# Project Memory

- If the H200 workspace is stopped, run `mygpu resume` and wait about 5 minutes
  before treating pod access as blocked.
- On `dev2`, the validated GeNN entrypoint is
  `/scratch/v1_l4_l23/smoke/genn/bin/genn-buildmodel.sh`, not
  `/scratch/genn-master-src/bin/genn-buildmodel.sh`.
- For this V1 SNN project, use GeNN/C++ as the default simulation stack. Do not
  propose PyTorch LIF prototypes unless the user explicitly asks for one.
- For SOM validation, require causal evidence that reducing SOM output changes
  broad-field L23E suppression; do not claim success from SOM firing alone.
- For project reports, explicitly document failed attempts, rejected probes,
  hardcoded assumptions, learned/plastic mechanisms, limited-emergent claims,
  engineering approximations, and unsupported claims. Do not treat architecture
  plus metrics as sufficient documentation.
- For multi-agent tasks in this repo, the lead must not start implementing while
  agents are pending. Task the fixed researcher/debugger/coder/validator team
  with concrete ownership first, wait for their outputs, then coordinate.
- If agents are running, do not return an idle/final placeholder. Keep waiting
  on agents, re-task them when needed, and only report once there is a concrete
  implementation, validation result, or blocker with evidence.
- For the higher-area stage, validate a standalone predictor first: lower V1
  activity should drive an HVA-like module that predicts future lower-V1 state
  from natural video. Do not add HVA-to-V1 feedback, VIP/SOM feedback, or
  modulatory currents until the predictor itself passes held-out prediction and
  anti-cheating controls.
- For the HVA predictor milestone, do not change the scientific target to easier
  channels when L2/3 is sparse. The required goal is future L2/3 activity
  prediction from L2/3 history. Improve methods for sparse L2/3 prediction
  (event/hazard objectives, train-only responsive selection, causal history,
  population-state metrics) rather than substituting L4/PV/SOM targets as
  pass/fail criteria.
- After lower-V1 video consolidation, event-timing validation must reset trial
  dynamic state before consolidation-enabled event trials so replay or
  consolidation membrane/synaptic transients cannot contaminate onset assays.
- Do not present softened population metrics as satisfying a user's raw
  exact-prediction target; explicitly separate exact raw recall from
  population-shape diagnostics.
- When the user asks for biological interpretation or remedy, answer the
  biological/computational point directly first. Do not lead with validator
  logs, task plumbing, or raw command-style summaries unless explicitly asked.
- For the next lower-V1 readiness milestone, L2/3 natural-video decodability
  below 0.6 is not acceptable. Do not present sub-0.6 improvements as success;
  improve the L2/3 representation itself while preserving biological gates.
- For this two-layer sensory V1 model, do not call ordinary video exposure or
  frozen evaluation "replay" or "consolidation" in user-facing explanations.
  Use "natural-video plastic exposure" and "frozen natural-video evaluation"
  unless a real hippocampal/replay mechanism is explicitly implemented.
- Do not conflate frame decoding with exact L2/3 activity prediction. Frame
  top-k asks whether stimulus identity is recoverable from the population;
  raw activity top-k asks whether the same L2/3 sites/tiles win across repeats.
  For the activity-reliability milestone, optimize and report raw top-k oracle
  / repeat stability first, with frame decoding only as a companion metric.
- Prefer the local RTX 5090 for GeNN validation and training runs when it is
  available, using `CUDA_VISIBLE_DEVICES=0`; do not kill unrelated GPU or pod
  processes while working.
- H200 pod workspace `dev-codex` is available via
  `MYGPU_WS=dev-codex mygpu connect`. Keep this project's files separated from
  other projects on shared drives. If any `runai`/`mygpu` command fails with
  token-expiry/login-required symptoms, run `bash ~/sih-gpu/app-login.sh` or
  `mygpu re-login`; never run plain `runai login` because it opens an
  interactive browser flow and can hang.
- For rate/homeostasis remediation, documenting an unresolved biological caveat
  is not completion when the user explicitly asks to keep solving it. Continue
  iterating on biologically plausible mechanisms until the stated L4/PV goals
  pass, or until there is a proven blocker accepted by the user.
- On branch `v1-biophysical-l23ee-plasticity`, the accepted replacement for the
  old top-k L23EE recurrent competition is documented in
  `docs/l23ee_triplet_homeostatic_plasticity_report.md`: old top-k disabled,
  triplet/homeostatic L23EE enabled with LR `1.25`, heldout fraction `0.15625`
  giving 54 video exposure frames, and strict acceptance log
  `.runs/logs/v1_triplet_l23ee_lr125_hold15625_acceptance_strict_validator.log`.
- Restart checkpoint 2026-06-09: branch `v1-biophysical-l23ee-plasticity` is
  clean and pushed to `neuroips` at commit
  `5679cf383e1380aacf701cb426253e0ccaa43f2d`. The next planned work is to
  evaluate and/or implement stable L2/3 population-representation and decoded
  future-state metrics for the higher predictor, using population vector
  correlation, RSM stability, decoder transfer, and future-state correlation as
  biology-aligned complements to raw exact top-k activity recall.
- For biology/neuroscience literature work on `reuben-ML`, first read
  `~/research-auth/README.md`. Prefer OA sources first (arXiv, PubMed Central,
  Unpaywall). For individual publisher papers, use
  `~/research-auth/fetch-paper <url> [outfile]`; use agentify-desktop MCP
  browser for JS-heavy/bot-walled pages, ask the user to perform SSO login if
  required, export cookies with `python3 ~/research-auth/export-cookies.py`,
  and never mass-crawl PDFs through EZproxy. Use Elsevier TDM API for
  ScienceDirect/Elsevier content.
- For neuroscience literature research on `reuben-ML`, use the USyd campus-IP
  access workflow in `~/research-auth/README.md`: try OA sources first
  (arXiv/PubMed Central/Unpaywall/OpenAlex), then use
  `~/research-auth/fetch-paper <url> [outfile]` for individual papers. For
  JS-heavy or bot-walled pages use the agentify browser; if SSO is required,
  ask the user to log in once and export cookies with
  `python3 ~/research-auth/export-cookies.py <domain>`. Never mass-crawl PDFs
  through EZproxy or publisher sites; bulk full text must use publisher TDM
  APIs, and ScienceDirect/Elsevier content should use the Elsevier TDM API.
