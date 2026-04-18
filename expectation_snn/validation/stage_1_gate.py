"""Stage 1 gate -- stability and learnability ONLY.

Checks:
- H recurrent bumps stable: leader/rotation-evoked bump persists 200-500 ms after input offset.
- Richter transition statistics measurable in H_R: MI(leader_t, H_state_{t+500ms}) > 0.
- Tang rotation structure measurable in H_T: similar MI over rotating regularity.
- No runaway, no silent H. 3 seeds sign-consistent.

NO Tang/Kok/Richter phenomenon checked here (that is the role of held-out assays).
"""
