"""Primary + secondary metrics.

Primary (plan sec 5):
1. Suppression vs preference for presented stimulus.
2. Suppression vs distance from expected feature.
3. Total V1 E population activity during probe epoch.
4. Preferred-channel gain (peak tuning-curve amplitude per cell at its theta_pref).
5. Tuning gain and width (width is explicitly secondary; Tang FWHM-null pre-registered).
6. Omission-subtracted sensory response (Kok only).

Secondary (comparability bridge):
- Cross-validated linear SVM decoding.
- 6-model pseudo-voxel forward family matched to Richter 2022
  (gain vs tuning, crossed with local/remote/global).

Signatures require Validator sign-off before Stage 1 implementation (plan sec 14.4, task #18).
"""
