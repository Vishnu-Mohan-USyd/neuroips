"""Sanity tests for :mod:`scripts.v2.eval_fmri_decoder`.

Three contracts the Task #74 fMRI-style decoder must satisfy, per the
dispatch spec:

1. **Retinotopy pooling mask**: each of the 256 L2/3 E units belongs
   to exactly one pseudo-voxel; the union covers the full population;
   every voxel has the same number of units (256 / n_voxels).
2. **Shuffle-label control**: when class labels are shuffled while the
   feature tensor is held fixed, 5-fold CV accuracy collapses to
   chance — proving the scoring pipeline doesn't leak any signal
   through a bug (fold split bias, stratification quirk, etc.).
3. **Determinism**: two runs of ``svm_5fold_cv_with_C`` on identical
   inputs with the same seed return bit-exact fold accuracies.

Tests 2 and 3 operate on a synthetic Gaussian-blob-per-class dataset
instead of a real forward pass — the pipeline's
data-collection-vs-classifier boundary is clean, and we want these
tests to be fast and deterministic in CI. The retinotopy test is a
pure numerical property of the pooling mask and has no runtime cost.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.v2.eval_fmri_decoder import (
    N_L23_E, N_RETINO_CELLS, N_ORIENT_BINS, SUPPORTED_N_VOXELS,
    build_voxel_pool_mask, pool_to_voxels, svm_5fold_cv_with_C,
)


# ---------------------------------------------------------------------------
# 1. Retinotopy pooling mask invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_voxels", SUPPORTED_N_VOXELS)
def test_voxel_mask_disjoint_union_and_equal_block_size(n_voxels):
    """Mask partitions the 256 L2/3 E units into equal-size voxels."""
    mask = build_voxel_pool_mask(n_voxels)
    assert mask.shape == (n_voxels, N_L23_E), (
        f"mask shape wrong: {mask.shape}"
    )
    assert mask.dtype == np.float32
    # Every unit belongs to EXACTLY one voxel (column sums == 1).
    col_sums = mask.sum(axis=0)
    assert np.array_equal(col_sums, np.ones(N_L23_E, dtype=np.float32)), (
        f"unit-to-voxel is not a partition: col_sums unique = "
        f"{np.unique(col_sums)}"
    )
    # Every voxel has the same number of units.
    expected_block = N_L23_E // n_voxels
    row_sums = mask.sum(axis=1)
    assert np.all(row_sums == expected_block), (
        f"voxel block size unequal: row_sums unique = {np.unique(row_sums)}"
    )
    # Union covers every unit.
    assert int(mask.sum()) == N_L23_E


def test_voxel_mask_respects_retinotopy():
    """Each voxel is contained within a single retinotopic cell.

    Units are tiled as retino_flat = i // 16, orient_bin = i % 16.
    A voxel must not straddle two retino cells — otherwise the
    "spatially localised Kok-style voxel" invariant is broken.
    """
    for n_voxels in SUPPORTED_N_VOXELS:
        mask = build_voxel_pool_mask(n_voxels)
        for v in range(n_voxels):
            units_in_v = np.flatnonzero(mask[v] > 0.5)
            retino_flats = units_in_v // N_ORIENT_BINS
            assert len(np.unique(retino_flats)) == 1, (
                f"voxel {v} at n_voxels={n_voxels} straddles retino cells "
                f"{set(retino_flats.tolist())}"
            )


def test_voxel_mask_rejects_unsupported_n_voxels():
    """Mask builder must refuse voxel counts that don't partition 16×16."""
    with pytest.raises(ValueError, match="n_voxels"):
        build_voxel_pool_mask(1)          # doesn't cover 16 retino cells
    with pytest.raises(ValueError, match="n_voxels"):
        build_voxel_pool_mask(48)         # 48/16=3; 16%3 != 0
    with pytest.raises(ValueError, match="n_voxels"):
        build_voxel_pool_mask(500)        # > 256


def test_pool_to_voxels_mean_pools_correctly():
    """``pool_to_voxels`` returns the within-voxel trial mean rate."""
    mask = build_voxel_pool_mask(16)      # one voxel per retino cell, 16 units each
    # Synthetic: voxel k has units with value k.
    r_l23 = np.zeros((3, N_L23_E), dtype=np.float32)
    for v in range(16):
        units = np.flatnonzero(mask[v] > 0.5)
        r_l23[:, units] = float(v)
    voxels = pool_to_voxels(r_l23, mask)
    expected = np.tile(np.arange(16, dtype=np.float32)[None, :], (3, 1))
    assert voxels.shape == (3, 16)
    np.testing.assert_allclose(voxels, expected, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Synthetic signal + scoring helpers
# ---------------------------------------------------------------------------


def _make_gaussian_blob_dataset(
    n_classes: int, n_per_class: int, n_features: int,
    *, separation: float, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)`` with class-conditional Gaussian blobs.

    Class means are placed along axis-aligned directions separated by
    ``separation`` standard deviations, so a linear classifier can
    achieve well above chance — this is what the shuffle-label control
    collapses. ``separation=5.0`` + ``n_per_class≥100`` puts the
    un-shuffled accuracy at ~1.0.
    """
    rng = np.random.default_rng(int(seed))
    centres = np.eye(max(n_classes, n_features), dtype=np.float32)
    centres = centres[:n_classes, :n_features] * float(separation)
    X_list = []
    y_list = []
    for c in range(n_classes):
        Xc = rng.standard_normal(
            (int(n_per_class), int(n_features)),
        ).astype(np.float32) + centres[c]
        X_list.append(Xc)
        y_list.extend([c] * int(n_per_class))
    X = np.concatenate(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)
    # Permute so class ordering is not correlated with row order.
    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# 2. Shuffle-label control
# ---------------------------------------------------------------------------


def test_shuffle_label_control_collapses_accuracy_to_chance():
    """With shuffled labels, 5-fold CV accuracy must hit chance ± 2σ.

    The feature tensor carries real class-conditional signal — so a
    normal fit reaches near-perfect accuracy; this test verifies that
    permuting the labels decouples signal from supervision and forces
    the classifier down to chance. If this ever fails the scoring
    pipeline is leaking labels through (e.g. stratification, inner CV
    seed reuse).
    """
    n_classes = 4
    n_per_class = 120
    X, y = _make_gaussian_blob_dataset(
        n_classes=n_classes, n_per_class=n_per_class,
        n_features=16, separation=5.0, seed=7,
    )
    # Un-shuffled sanity: near-perfect accuracy (proves the dataset
    # has real, decodable signal before we shuffle it).
    real = svm_5fold_cv_with_C(X, y, seed=123)
    assert "error" not in real, real
    assert real["acc_mean"] > 0.90, (
        f"synthetic-signal dataset should decode easily, but "
        f"acc_mean={real['acc_mean']:.4f} — test precondition broken"
    )

    # Shuffle labels (permute a COPY so X is untouched) and re-run.
    rng = np.random.default_rng(99)
    y_shuf = y.copy()
    rng.shuffle(y_shuf)
    shuffled = svm_5fold_cv_with_C(X, y_shuf, seed=123)
    assert "error" not in shuffled, shuffled

    chance = 1.0 / n_classes
    # 2σ tolerance of fold-acc std. Under a shuffled 4-class problem
    # at N_total=480 with ~96 samples per fold, the fold accuracy
    # binomial std is sqrt(p(1-p)/n) ≈ 0.044, so 2σ ≈ 0.088. The
    # per-fold SVM is at most this noisy.
    tol = max(2.0 * float(shuffled["acc_std"]), 0.10)
    assert abs(shuffled["acc_mean"] - chance) < tol, (
        f"shuffled-label accuracy={shuffled['acc_mean']:.4f} is "
        f"further than 2σ from chance={chance:.4f} (std={shuffled['acc_std']:.4f}); "
        f"pipeline may be leaking labels"
    )


# ---------------------------------------------------------------------------
# 3. Determinism
# ---------------------------------------------------------------------------


def test_svm_cv_is_deterministic_under_same_seed():
    """Two calls with identical inputs + seed must give bit-exact acc."""
    X, y = _make_gaussian_blob_dataset(
        n_classes=3, n_per_class=80, n_features=10,
        separation=4.0, seed=42,
    )
    r1 = svm_5fold_cv_with_C(X, y, seed=999)
    r2 = svm_5fold_cv_with_C(X, y, seed=999)
    assert "error" not in r1 and "error" not in r2
    assert r1["acc_mean"] == r2["acc_mean"], (
        f"bit-exact determinism violated: {r1['acc_mean']} vs "
        f"{r2['acc_mean']}"
    )
    assert r1["per_fold_acc"] == r2["per_fold_acc"], (
        f"per-fold accuracies differ across runs: "
        f"{r1['per_fold_acc']} vs {r2['per_fold_acc']}"
    )
    assert r1["per_fold_C"] == r2["per_fold_C"]


def test_svm_cv_seed_change_changes_folds():
    """Sanity: different seed perturbs fold splits and therefore accuracy.

    Guards against a bug where the seed is ignored and every run
    happens to return the same numbers — which would make the
    determinism test above trivially pass but offer no real guarantee.
    """
    X, y = _make_gaussian_blob_dataset(
        n_classes=3, n_per_class=80, n_features=10,
        separation=1.5, seed=1,  # lower separation → more fold-split sensitivity
    )
    r_a = svm_5fold_cv_with_C(X, y, seed=1)
    r_b = svm_5fold_cv_with_C(X, y, seed=2)
    # Per-fold accuracies must not ALL be identical.
    assert r_a["per_fold_acc"] != r_b["per_fold_acc"], (
        "changing the CV seed did not change per-fold accuracy — "
        "the random_state is probably not being threaded into "
        "StratifiedKFold"
    )


# ---------------------------------------------------------------------------
# 4. cue_mode routing (Δdecode extension, Task #74)
# ---------------------------------------------------------------------------


def test_cue_mode_expected_differs_from_none_on_nonunity_gain():
    """With non-unity ``W_q_gain`` and non-zero ``W_qm_task``, running
    one trial at ``cue_mode='expected'`` must differ from the same trial
    at ``cue_mode='none'``.

    ``cue_mode='none'`` delivers ``q_t=None`` every step, bypassing both
    β pathways (``l23_e.W_q_gain`` at ``layers.py:680`` and
    ``context_memory.W_qm_task`` at ``context_memory.py:421``).
    ``cue_mode='expected'`` delivers ``q_t=q_cue`` every step, engaging
    both. A β-trained ckpt has learned values on both weights, so the
    per-trial L2/3 E rate tensors MUST differ or the cue_mode routing
    has no numerical effect — i.e. the Δdecode protocol is measuring
    nothing.

    We don't load a real β ckpt here — we build a fresh V2Network and
    install a synthetic non-unity ``W_q_gain`` (deterministic, in-memory)
    to exercise the routing without a multi-minute training run.
    """
    from scripts.v2._gates_common import CheckpointBundle
    from scripts.v2.eval_fmri_decoder import run_trial
    from src.v2_model.config import ModelConfig
    from src.v2_model.network import V2Network
    from src.v2_model.stimuli.feature_tokens import TokenBank
    import torch

    cfg = ModelConfig(seed=42, device="cpu")
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=42, device="cpu")
    net.set_phase("phase3_kok")
    net.eval()
    bundle = CheckpointBundle(cfg=cfg, net=net, bank=bank, meta={})

    # Install non-unity W_q_gain for cue 0 (135° in the default
    # cue_mapping from seed 42). Without this, even with correct
    # cue_mode routing, the β gate is an identity and the test can't
    # distinguish it from cue_mode='none'.
    cue_id_expected = 0  # corresponds to 135° under cue_mapping_from_seed(42)
    torch.manual_seed(321)
    bundle.net.l23_e.W_q_gain[cue_id_expected] = (
        0.1 + 0.8 * torch.rand(bundle.net.l23_e.W_q_gain.shape[1])
    )

    g_none = torch.Generator().manual_seed(2026)
    r_none = run_trial(
        bundle, orientation_deg=135.0, contrast=0.9,
        noise_std=0.0, n_warmup=10, n_readout=5,
        generator=g_none, cue_id=None,
    )
    g_exp = torch.Generator().manual_seed(2026)
    r_exp = run_trial(
        bundle, orientation_deg=135.0, contrast=0.9,
        noise_std=0.0, n_warmup=10, n_readout=5,
        generator=g_exp, cue_id=cue_id_expected,
    )
    assert not torch.equal(r_none, r_exp), (
        "cue_mode='expected' produces identical rates to cue_mode='none' "
        "even with non-unity W_q_gain — the q_t=q_cue routing is not "
        "reaching the forward pass"
    )


# ---------------------------------------------------------------------------
# 5. Frozen-localizer null-control (Task #74 redesign)
# ---------------------------------------------------------------------------


def test_frozen_localizer_delta_is_null_when_beta_disabled():
    """Null control: with ``W_q_gain=1.0`` everywhere and
    ``W_qm_task=0``, every cue-mode path is a mechanical no-op on the
    network, so the frozen-localizer protocol must yield
    ``acc_matched ≈ acc_mismatched ≈ localizer_acc`` within 2σ.

    Why zero ``W_qm_task``: a fresh ``V2Network`` has ``W_qm_task`` at
    its random-normal init (task_input_init_std), so even with
    ``W_q_gain`` at 1.0 the cue still drives ``context_memory`` via
    :file:`src/v2_model/context_memory.py:421`, breaking the null
    expected by the lead's spec ("β disabled → no cue effect"). We
    therefore disable BOTH β pathways; what remains is the pure
    decoder variance from finite trial counts, which the 2σ tolerance
    accommodates.

    We build a tiny in-memory bundle (no ckpt fixture), run the
    protocol with noise=0 + high contrast so the localizer decoder
    reliably learns 45°/135° from few trials, and use trial counts
    large enough for 5-fold inner CV (≥5/class → ≥10/class to give
    inner/outer room).
    """
    from scripts.v2._gates_common import CheckpointBundle
    from scripts.v2.eval_fmri_decoder import build_voxel_pool_mask
    from scripts.v2.eval_fmri_decoder_kok import (
        run_frozen_localizer_protocol,
    )
    from src.v2_model.config import ModelConfig
    from src.v2_model.network import V2Network
    from src.v2_model.stimuli.feature_tokens import TokenBank
    import torch

    cfg = ModelConfig(seed=42, device="cpu")
    bank = TokenBank(cfg, seed=0)
    net = V2Network(cfg, token_bank=bank, seed=42, device="cpu")
    net.set_phase("phase3_kok")
    net.eval()

    # Disable both β pathways: W_q_gain at its all-1.0 init AND zero
    # the context_memory W_qm_task so the cue cannot reach m_t. This
    # is the "β disabled" null condition the spec asks for.
    assert torch.all(net.l23_e.W_q_gain == 1.0)
    with torch.no_grad():
        net.context_memory.W_qm_task.zero_()

    bundle = CheckpointBundle(cfg=cfg, net=net, bank=bank, meta={})
    mask = build_voxel_pool_mask(64)
    cue_mapping = {0: 135.0, 1: 45.0}  # matches cue_mapping_from_seed(42)

    result = run_frozen_localizer_protocol(
        bundle, cue_mapping=cue_mapping, mask=mask,
        # Enough samples for 5-fold inner CV (≥5/class) with margin
        # for stable fold accuracies; kept small to cap wall-clock.
        n_trials_localizer_per_orient=40,
        n_trials_test_per_orient=40,
        # Zero noise + high contrast → localizer gets an easy signal
        # and can actually learn from few trials.
        noise_std=0.0,
        contrast_min=0.9, contrast_max=1.0,
        n_warmup=15, n_readout=5,
        trial_seed=11, cv_seed=11,
        n_bootstrap=200, n_permutations=200,
        verdict_eps=0.02,
    )

    loc_acc = result["localizer"]["cv_acc_mean"]
    acc_m = result["cued_matched"]["acc"]
    acc_u = result["cued_mismatched"]["acc"]
    delta = result["delta_matched_minus_mismatched"]["mean"]

    # The localizer must actually learn something — otherwise "Δ ≈ 0"
    # is a trivial pass. Under zero noise + contrast=1 a fresh V2Network
    # with the Fix-K orientation-biased L4→L23 mask produces a cleanly
    # orientation-tuned L2/3 response.
    assert loc_acc > 0.75, (
        f"localizer decoder is not learning 45°/135° under zero noise "
        f"(loc_acc={loc_acc:.4f}); test premise broken — can't assess "
        f"Δ equality against a chance-level decoder"
    )

    # Null-control claim: with both β pathways disabled the matched and
    # mismatched test sets are sampled from the SAME distribution (cue_id
    # is numerically inert), so Δ must be within binomial noise of 0.
    # At n=80/condition with p≈loc_acc, 1σ ≈ sqrt(p(1-p)/n) ≲ 0.06, so
    # 2σ ≲ 0.12. We use 0.15 as a safety margin.
    assert abs(delta) < 0.15, (
        f"Δ={delta:+.4f} exceeds 2σ null bound when β is disabled "
        f"(loc={loc_acc:.3f}, acc_m={acc_m:.3f}, acc_u={acc_u:.3f}) "
        f"— either a β pathway is still active, or trial variance is "
        f"larger than expected"
    )
    # Both accs should also land near the localizer accuracy.
    for tag, val in (("acc_matched", acc_m), ("acc_mismatched", acc_u)):
        assert abs(val - loc_acc) < 0.20, (
            f"{tag}={val:.3f} drifts too far from loc_acc={loc_acc:.3f} "
            f"under β-disabled null control"
        )
    # And the emitted verdict must be null by construction.
    assert result["verdict"] == "null", (
        f"β-disabled null control should verdict=null, got "
        f"{result['verdict']!r} (Δ={delta:+.4f})"
    )
