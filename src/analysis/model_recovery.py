"""Model recovery / sensitivity analysis.

Generates synthetic ground-truth L2/3 responses for each feedback mechanism
using hand-constructed parameters (no training needed), then verifies that
the analysis pipeline correctly identifies the planted mechanism.

Key components:
    1. Synthetic response generation (analytical, from mechanism equations)
    2. Suppression-by-tuning profile extraction
    3. Parametric model fitting (Gaussian trough, Mexican hat DoG, offset Gaussian)
    4. Observation model (L2/3 -> synthetic voxels + noise)
    5. MVPA classification (expected vs unexpected)

Mechanism-to-profile mapping (derived from the physics):
    - Dampening (A): SOM peaks at expected → Gaussian trough at center
    - Sharpening (B): DoG SOM → Mexican hat (center peak, flanks dip, recovery to 0)
    - Center-surround (C): broad SOM + center excitation → offset Gaussian
      (center peak + persistent negative baseline at all flanks)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from src.config import ModelConfig, MechanismType
from src.model.network import LaminarV1V2Network
from src.stimulus.gratings import generate_grating
from src.utils import circular_distance_abs


# ---------------------------------------------------------------------------
# 1. Synthetic ground-truth response generation
# ---------------------------------------------------------------------------

@dataclass
class SyntheticResponses:
    """Synthetic L2/3 responses for one mechanism under expected/unexpected/neutral."""
    mechanism: MechanismType
    expected: Tensor      # [n_units] L2/3 response when stimulus = expected orientation
    unexpected: Tensor    # [n_units] L2/3 response when stimulus = unexpected orientation
    neutral: Tensor       # [n_units] L2/3 response when stimulus = neutral (no prediction)
    pref_oris: Tensor     # [n_units] preferred orientation of each unit (degrees)
    expected_ori: float   # expected orientation (degrees)
    unexpected_ori: float # unexpected orientation (degrees)


def generate_synthetic_responses(
    mechanism: MechanismType,
    expected_ori: float = 45.0,
    unexpected_ori: float = 90.0,
    contrast: float = 0.8,
    pi_pred: float = 3.0,
    n_orientations: int = 36,
    n_timesteps: int = 30,
    seed: int = 42,
) -> SyntheticResponses:
    """Generate synthetic L2/3 responses by running the full network with
    hand-set V2 outputs (bypassing V2 learning).

    For expected: q_pred peaks at expected_ori, stimulus at expected_ori
    For unexpected: q_pred peaks at expected_ori, stimulus at unexpected_ori
    For neutral: pi_pred = 0 (no prediction), stimulus at expected_ori

    Returns:
        SyntheticResponses with L2/3 steady-state responses for each condition.
    """
    torch.manual_seed(seed)
    N = n_orientations
    cfg = ModelConfig(mechanism=mechanism, n_orientations=N, feedback_mode='fixed')
    net = LaminarV1V2Network(cfg)
    net.eval()

    period = cfg.orientation_range
    step = period / N
    pref_oris = torch.arange(N, dtype=torch.float32) * step

    # Expected channel for q_pred
    exp_channel = int(round(expected_ori / step)) % N
    q_pred = torch.zeros(1, N)
    q_pred[0, exp_channel] = 1.0

    results = {}
    conditions = {
        "expected": (expected_ori, pi_pred),
        "unexpected": (unexpected_ori, pi_pred),
        "neutral": (expected_ori, 0.0),
    }

    with torch.no_grad():
        for cond_name, (stim_ori, pi) in conditions.items():
            stim = generate_grating(
                torch.tensor([stim_ori]),
                torch.tensor([contrast]),
                n_orientations=N, sigma=cfg.sigma_ff,
                n=cfg.naka_rushton_n, c50=cfg.naka_rushton_c50,
                period=period,
            )
            r_l23 = _run_with_fixed_v2(
                net, stim, q_pred, torch.tensor([[pi]]),
                n_timesteps=n_timesteps,
            )
            results[cond_name] = r_l23.squeeze(0)

    return SyntheticResponses(
        mechanism=mechanism,
        expected=results["expected"],
        unexpected=results["unexpected"],
        neutral=results["neutral"],
        pref_oris=pref_oris,
        expected_ori=expected_ori,
        unexpected_ori=unexpected_ori,
    )


def _run_with_fixed_v2(
    net: LaminarV1V2Network,
    stimulus: Tensor,
    q_pred: Tensor,
    pi_pred: Tensor,
    n_timesteps: int = 30,
) -> Tensor:
    """Run the network with fixed V2 outputs, bypassing V2 module."""
    B, N = stimulus.shape
    device = stimulus.device

    r_l4 = torch.zeros(B, N, device=device)
    r_l23 = torch.zeros(B, N, device=device)
    r_pv = torch.zeros(B, 1, device=device)
    r_som = torch.zeros(B, N, device=device)
    adaptation = torch.zeros(B, N, device=device)

    for _ in range(n_timesteps):
        r_l4, adaptation = net.l4(stimulus, r_l4, r_pv, adaptation)
        r_pv = net.pv(r_l4, r_l23, r_pv)
        deep_tmpl = net.deep_template(q_pred, pi_pred)
        som_drive = net.feedback.compute_som_drive(q_pred, pi_pred)
        r_som = net.som(som_drive, r_som)
        template_mod = net.feedback.compute_center_excitation(q_pred, pi_pred)
        l4_to_l23 = net.feedback.compute_error_signal(r_l4, deep_tmpl)
        r_l23 = net.l23(l4_to_l23, r_l23, template_mod, r_som, r_pv)

    return r_l23


# ---------------------------------------------------------------------------
# 2. Suppression-by-tuning profile
# ---------------------------------------------------------------------------

@dataclass
class SuppressionProfile:
    """Suppression and surprise profiles as function of tuning distance."""
    delta_theta: Tensor      # [n_bins] angular distances from expected (degrees)
    suppression: Tensor      # [n_bins] Δ_supp = response(expected) - response(neutral)
    surprise: Tensor         # [n_bins] Δ_surp = response(unexpected) - response(neutral)
    raw_expected: Tensor     # [n_bins] mean response in expected condition
    raw_unexpected: Tensor   # [n_bins] mean response in unexpected condition
    raw_neutral: Tensor      # [n_bins] mean response in neutral condition


def compute_suppression_profile(
    responses: SyntheticResponses,
    period: float = 180.0,
) -> SuppressionProfile:
    """Compute suppression-by-tuning and surprise-by-tuning profiles.

    Bins L2/3 units by |pref_θ - expected_θ|. Averages symmetric units
    (same |distance|) to get one value per unique distance.
    """
    pref = responses.pref_oris
    exp_ori = responses.expected_ori

    dists = circular_distance_abs(pref, torch.tensor(exp_ori), period=period)

    # Average symmetric units at same distance
    unique_dists = dists.unique(sorted=True)
    avg_expected = []
    avg_unexpected = []
    avg_neutral = []

    for d in unique_dists:
        mask = (dists - d).abs() < 0.1
        avg_expected.append(responses.expected[mask].mean())
        avg_unexpected.append(responses.unexpected[mask].mean())
        avg_neutral.append(responses.neutral[mask].mean())

    avg_expected = torch.stack(avg_expected)
    avg_unexpected = torch.stack(avg_unexpected)
    avg_neutral = torch.stack(avg_neutral)

    suppression = avg_expected - avg_neutral
    surprise = avg_unexpected - avg_neutral

    return SuppressionProfile(
        delta_theta=unique_dists,
        suppression=suppression,
        surprise=surprise,
        raw_expected=avg_expected,
        raw_unexpected=avg_unexpected,
        raw_neutral=avg_neutral,
    )


# ---------------------------------------------------------------------------
# 3. Parametric model fitting
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """Result of fitting a parametric model to a suppression profile."""
    model_name: str       # "gaussian_trough", "mexican_hat", "offset_gaussian"
    r_squared: float      # R² goodness of fit
    params: dict          # Fitted parameters
    fitted_curve: Tensor  # [n_points] fitted values


def _r_squared(y_true: Tensor, y_pred: Tensor) -> float:
    """Compute R² (coefficient of determination)."""
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if ss_tot < 1e-12:
        return 0.0
    return (1.0 - ss_res / ss_tot).item()


def fit_parametric_models(
    profile: SuppressionProfile,
) -> list[FitResult]:
    """Fit three parametric models to the normalized suppression profile.

    Models (derived from mechanism physics):
        1. gaussian_trough (dampening): A * exp(-x²/(2σ²)), A < 0
           → Trough at center, recovers to 0 at flanks.
        2. mexican_hat (sharpening): A_narrow * exp(-x²/(2σ_n²)) + A_broad * exp(-x²/(2σ_b²))
           with A_narrow > 0, A_broad < 0, σ_b > σ_n
           → Peak at center, dip at intermediate, recovers to 0 far away.
        3. offset_gaussian (center-surround): C + A * exp(-x²/(2σ²)), C < 0, A > 0
           → Constant baseline suppression + narrow center peak.
           Critically: does NOT return to 0 at large distances.

    Normalizes the profile to [-1, 1] for grid search, then rescales.

    Returns:
        List of 3 FitResult objects.
    """
    x = profile.delta_theta
    y = profile.suppression

    # Normalize for grid search
    y_range = max(y.abs().max().item(), 1e-6)
    y_norm = y / y_range

    results = []

    # 1. Gaussian trough (dampening): y = A * exp(-x²/(2σ²))
    best_r2 = -float("inf")
    best_params = {}
    best_curve = torch.zeros_like(y_norm)
    for a in [v * 0.05 for v in range(-20, 1)]:  # A ∈ [-1.0, 0.0]
        for sigma in [5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]:
            curve = a * torch.exp(-x ** 2 / (2 * sigma ** 2))
            r2 = _r_squared(y_norm, curve)
            if r2 > best_r2:
                best_r2 = r2
                best_params = {"amplitude": a * y_range, "sigma": sigma}
                best_curve = curve * y_range
    results.append(FitResult("gaussian_trough", best_r2, best_params, best_curve))

    # 2. Mexican hat (sharpening): y = A_n * exp(-x²/(2σ_n²)) + A_b * exp(-x²/(2σ_b²))
    #    Constraint: A_n > 0, A_b < 0, σ_b > σ_n (narrow peak + broad dip, returns to 0)
    best_r2 = -float("inf")
    best_params = {}
    best_curve = torch.zeros_like(y_norm)
    for a_n in [v * 0.1 for v in range(1, 16)]:  # A_narrow ∈ [0.1, 1.5]
        for s_n in [5.0, 8.0, 10.0, 12.0, 15.0]:
            for a_b in [v * 0.1 for v in range(-10, 0)]:  # A_broad ∈ [-1.0, -0.1]
                for s_b in [20.0, 25.0, 30.0, 40.0, 50.0, 60.0]:
                    if s_b <= s_n:
                        continue
                    curve = (a_n * torch.exp(-x ** 2 / (2 * s_n ** 2))
                             + a_b * torch.exp(-x ** 2 / (2 * s_b ** 2)))
                    r2 = _r_squared(y_norm, curve)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_params = {
                            "a_narrow": a_n * y_range, "sigma_narrow": s_n,
                            "a_broad": a_b * y_range, "sigma_broad": s_b,
                        }
                        best_curve = curve * y_range
    results.append(FitResult("mexican_hat", best_r2, best_params, best_curve))

    # 3. Offset Gaussian (center-surround): y = C + A * exp(-x²/(2σ²))
    #    Constraint: C < 0 (persistent baseline), A > 0 (center peak)
    #    Key: y → C (nonzero) at large distances — distinguishes from mexican_hat
    best_r2 = -float("inf")
    best_params = {}
    best_curve = torch.zeros_like(y_norm)
    for c in [v * 0.05 for v in range(-20, 1)]:  # C ∈ [-1.0, 0.0]
        for a in [v * 0.1 for v in range(1, 21)]:  # A ∈ [0.1, 2.0]
            for sigma in [5.0, 8.0, 10.0, 12.0, 15.0, 20.0]:
                curve = c + a * torch.exp(-x ** 2 / (2 * sigma ** 2))
                r2 = _r_squared(y_norm, curve)
                if r2 > best_r2:
                    best_r2 = r2
                    best_params = {
                        "offset": c * y_range, "amplitude": a * y_range, "sigma": sigma,
                    }
                    best_curve = curve * y_range
    results.append(FitResult("offset_gaussian", best_r2, best_params, best_curve))

    return results


def identify_mechanism(fit_results: list[FitResult]) -> str:
    """Identify which mechanism best explains the suppression profile.

    Returns the model_name with the highest R².
    """
    best = max(fit_results, key=lambda f: f.r_squared)
    return best.model_name


# ---------------------------------------------------------------------------
# 4. Observation model: L2/3 -> synthetic voxels
# ---------------------------------------------------------------------------

@dataclass
class VoxelResponses:
    """Synthetic voxel responses from pooled L2/3 activity."""
    expected: Tensor     # [n_voxels]
    unexpected: Tensor   # [n_voxels]
    neutral: Tensor      # [n_voxels]
    n_voxels: int
    snr: float


def make_observation_model(
    responses: SyntheticResponses,
    n_voxels: int = 8,
    snr: float = 5.0,
    n_trials: int = 100,
    seed: int = 42,
) -> tuple[VoxelResponses, Tensor, Tensor]:
    """Pool L2/3 into synthetic voxels with noise.

    Each voxel pools a contiguous block of L2/3 units.
    Gaussian noise added at SNR level.

    Returns:
        voxel_responses: Mean voxel responses per condition.
        trial_patterns: [n_trials * 3, n_voxels] noisy trial-level patterns.
        trial_labels: [n_trials * 3] condition labels (0=expected, 1=unexpected, 2=neutral).
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    N = responses.expected.shape[0]
    units_per_voxel = N // n_voxels

    # Pooling matrix: each voxel sums a contiguous block with slight overlap
    pool_weights = torch.zeros(n_voxels, N)
    for v in range(n_voxels):
        start = v * units_per_voxel
        end = min(start + units_per_voxel, N)
        pool_weights[v, start:end] = 1.0 / units_per_voxel

    conditions = {
        "expected": responses.expected,
        "unexpected": responses.unexpected,
        "neutral": responses.neutral,
    }

    pooled = {}
    for name, resp in conditions.items():
        pooled[name] = pool_weights @ resp

    # Generate noisy trials
    trial_patterns = []
    trial_labels = []
    for label, name in enumerate(["expected", "unexpected", "neutral"]):
        signal = pooled[name]
        noise_std = signal.abs().mean() / max(snr, 1e-6)
        for _ in range(n_trials):
            noise = torch.randn(n_voxels, generator=gen) * noise_std
            trial_patterns.append(signal + noise)
            trial_labels.append(label)

    trial_patterns = torch.stack(trial_patterns)
    trial_labels = torch.tensor(trial_labels)

    voxel_resp = VoxelResponses(
        expected=pooled["expected"],
        unexpected=pooled["unexpected"],
        neutral=pooled["neutral"],
        n_voxels=n_voxels,
        snr=snr,
    )

    return voxel_resp, trial_patterns, trial_labels


# ---------------------------------------------------------------------------
# 5. MVPA classification
# ---------------------------------------------------------------------------

def mvpa_classify(
    trial_patterns: Tensor,
    trial_labels: Tensor,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, float]:
    """Nearest-centroid MVPA classifier with cross-validation.

    Returns dict with 'acc_2way' (expected vs unexpected) and 'acc_3way'.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    n_trials = trial_patterns.shape[0]
    perm = torch.randperm(n_trials, generator=gen)
    fold_size = n_trials // n_folds

    correct_2way = 0
    total_2way = 0
    correct_3way = 0
    total_3way = 0

    for fold in range(n_folds):
        test_idx = perm[fold * fold_size:(fold + 1) * fold_size]
        train_idx = torch.cat([perm[:fold * fold_size], perm[(fold + 1) * fold_size:]])

        train_X = trial_patterns[train_idx]
        train_y = trial_labels[train_idx]
        test_X = trial_patterns[test_idx]
        test_y = trial_labels[test_idx]

        centroids = {}
        for c in range(3):
            mask = train_y == c
            if mask.sum() > 0:
                centroids[c] = train_X[mask].mean(dim=0)

        for i in range(len(test_X)):
            dists = {c: ((test_X[i] - centroids[c]) ** 2).sum() for c in centroids}

            pred_3way = min(dists, key=dists.get)
            if pred_3way == test_y[i].item():
                correct_3way += 1
            total_3way += 1

            if test_y[i].item() in (0, 1):
                dists_2way = {c: dists[c] for c in [0, 1] if c in dists}
                if dists_2way:
                    pred_2way = min(dists_2way, key=dists_2way.get)
                    if pred_2way == test_y[i].item():
                        correct_2way += 1
                    total_2way += 1

    return {
        "acc_2way": correct_2way / max(total_2way, 1),
        "acc_3way": correct_3way / max(total_3way, 1),
    }


# ---------------------------------------------------------------------------
# 6. Full recovery pipeline
# ---------------------------------------------------------------------------

@dataclass
class RecoveryResult:
    """Result of the full model recovery analysis for one mechanism."""
    mechanism: MechanismType
    profile: SuppressionProfile
    fit_results: list[FitResult]
    identified_mechanism: str
    correctly_identified: bool
    voxel_results: dict[str, dict]


# Mapping from mechanism type to expected best-fit model name
# Derived from the actual physics of each mechanism's SOM/excitation profile:
# - Dampening: SOM peaks at expected → Gaussian trough
# - Sharpening: DoG SOM → Mexican hat (peak, dip, recovery to 0)
# - Center-surround: broad SOM + center excitation → offset Gaussian (persistent baseline)
MECHANISM_TO_FIT_MODEL = {
    MechanismType.DAMPENING: "gaussian_trough",
    MechanismType.SHARPENING: "mexican_hat",
    MechanismType.CENTER_SURROUND: "offset_gaussian",
}


def run_recovery(
    mechanism: MechanismType,
    voxel_counts: list[int] | None = None,
    snr_levels: list[float] | None = None,
    seed: int = 42,
) -> RecoveryResult:
    """Run the full model recovery pipeline for one mechanism.

    1. Generate synthetic responses
    2. Compute suppression profile (averaged over symmetric units)
    3. Fit parametric models
    4. Run observation model at multiple granularities/noise levels
    5. Check if mechanism is correctly identified
    """
    if voxel_counts is None:
        voxel_counts = [4, 8, 16]
    if snr_levels is None:
        snr_levels = [1.0, 5.0, 20.0]

    responses = generate_synthetic_responses(mechanism, seed=seed)
    profile = compute_suppression_profile(responses)
    fit_results = fit_parametric_models(profile)
    identified = identify_mechanism(fit_results)

    voxel_results = {}
    for n_vox in voxel_counts:
        for snr in snr_levels:
            key = f"vox{n_vox}_snr{snr}"
            _, patterns, labels = make_observation_model(
                responses, n_voxels=n_vox, snr=snr, seed=seed
            )
            mvpa = mvpa_classify(patterns, labels, seed=seed)
            voxel_results[key] = mvpa

    expected_model = MECHANISM_TO_FIT_MODEL.get(mechanism)
    correctly_identified = (identified == expected_model) if expected_model else True

    return RecoveryResult(
        mechanism=mechanism,
        profile=profile,
        fit_results=fit_results,
        identified_mechanism=identified,
        correctly_identified=correctly_identified,
        voxel_results=voxel_results,
    )
