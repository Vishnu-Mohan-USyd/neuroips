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
