"""Three-stage training drivers.

Stage 0: calibration (hard-wire V1 ring, tune PV iSTDP + bias + g_total).
Stage 1: incidental context learning on H recurrent E<->E only (H_R, H_T separately).
Stage 2: cue -> H_R via teacher-forced eligibility-trace learning; H recurrent frozen.
"""
