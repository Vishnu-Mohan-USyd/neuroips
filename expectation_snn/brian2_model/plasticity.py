"""Plasticity rules.

- Pair-STDP with multiplicative weight-dependent bounds + postsynaptic normalization
  (H recurrent E<->E, Stage 1).
- Vogels 2011 iSTDP (PV pool stabilizer, Stage 0).
- Eligibility-trace + teacher-forced rule for cue -> H_R (Stage 2, plan sec 2).
- Saponati & Vinck 2023 predictive rule reserved as fallback only if pair-STDP fails.
"""
