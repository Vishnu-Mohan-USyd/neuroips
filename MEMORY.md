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
