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
