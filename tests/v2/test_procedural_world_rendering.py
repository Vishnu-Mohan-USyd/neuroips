"""Rendering semantics: contrast, occluder, identity, rotation, position.

The rendering path (``affine_grid`` + ``grid_sample`` on ``token − 0.5``,
with zero padding and a post-hoc ``+ 0.5``) has a handful of contractual
invariants that the downstream circuit relies on:

* Contrast 0 gives a uniformly grey frame (``0.5`` everywhere). This is
  the Richter-like "blank frame" condition.
* Different identities with everything else fixed produce distinguishable
  frames — otherwise the identity axis carries no signal.
* The occluder paints a fixed 10×10 grey patch at the bottom-right when
  ``o=1``; it is otherwise a no-op.
* Frames have mean close to 0.5 (background grey dominates an off-center
  token with modest contrast).
"""

from __future__ import annotations

import pytest
import torch

from src.v2_model.stimuli.feature_tokens import TokenBank
from src.v2_model.world import ProceduralWorld, WorldState
from src.v2_model.world.procedural import (
    OCCLUDER_SIZE, OCCLUDER_X0, OCCLUDER_Y0, TARGET_MEAN,
)


@pytest.fixture(scope="module")
def bank():
    from src.v2_model.config import ModelConfig
    return TokenBank(ModelConfig(), seed=0)


@pytest.fixture
def world(cfg, bank):
    return ProceduralWorld(cfg, bank, seed_family="train")


def _state(**kw) -> WorldState:
    """Build a WorldState with sensible defaults + overrides."""
    defaults = dict(
        z=0, theta=0.0, position=(16, 16), contrast=1.0,
        occluder=0, regime="CW-drift", step_idx=0,
    )
    defaults.update(kw)
    return WorldState(**defaults)


def test_contrast_zero_gives_uniform_background(world: ProceduralWorld) -> None:
    """contrast=0 → ``frame`` is uniformly 0.5 (no identity information)."""
    for z in range(12):
        frame = world.render(_state(z=z, contrast=0.0))
        assert frame.shape == (1, 32, 32)
        assert torch.allclose(
            frame, torch.full_like(frame, TARGET_MEAN), atol=1e-6,
        ), f"contrast=0 frame has variation at z={z}: std={float(frame.std())}"


def test_different_z_give_distinguishable_frames(world: ProceduralWorld) -> None:
    """Holding all else equal, distinct z values → distinct frames."""
    f_prev = None
    for z in range(12):
        f = world.render(_state(z=z, contrast=1.0))
        if f_prev is not None:
            diff = float((f - f_prev).abs().mean())
            assert diff > 1e-3, (
                f"z={z} frame indistinguishable from z={z-1} (mean-abs diff "
                f"{diff:.6f})"
            )
        f_prev = f


def test_frames_are_non_trivial(world: ProceduralWorld) -> None:
    """Default-contrast frames have non-zero pixel variation."""
    frames, _ = world.trajectory(0, n_steps=30)
    for i in range(frames.shape[0]):
        assert float(frames[i].std()) > 1e-3, (
            f"frame {i} is uniformly grey (std={float(frames[i].std())})"
        )


def test_occluder_paints_fixed_corner(world: ProceduralWorld) -> None:
    """``o=1`` overrides the bottom-right 10×10 patch to 0.5 exactly."""
    f_clean = world.render(_state(occluder=0, contrast=1.0))
    f_occ = world.render(_state(occluder=1, contrast=1.0))
    patch_occ = f_occ[
        :, OCCLUDER_Y0:OCCLUDER_Y0 + OCCLUDER_SIZE,
        OCCLUDER_X0:OCCLUDER_X0 + OCCLUDER_SIZE,
    ]
    assert torch.allclose(
        patch_occ, torch.full_like(patch_occ, TARGET_MEAN), atol=1e-6,
    ), "occluder patch is not uniformly 0.5"
    # Outside-patch pixels should match the clean frame (rotation/translation
    # is unchanged by the occluder paint).
    mask = torch.ones_like(f_clean, dtype=torch.bool)
    mask[:, OCCLUDER_Y0:, OCCLUDER_X0:] = False
    torch.testing.assert_close(
        f_clean[mask], f_occ[mask], atol=0.0, rtol=0.0,
    )


def test_position_translates_content(world: ProceduralWorld) -> None:
    """Moving p shifts the non-grey content spatially."""
    # Use a token that is noticeably non-uniform, centered, no occluder.
    f_center = world.render(_state(position=(16, 16), contrast=1.0))
    f_edge = world.render(_state(position=(2, 2), contrast=1.0))
    # Total absolute deviation from 0.5 should be concentrated in different
    # spatial regions.
    dev_c = (f_center - TARGET_MEAN).abs()
    dev_e = (f_edge - TARGET_MEAN).abs()
    # Both frames carry content (non-zero deviation)
    assert float(dev_c.sum()) > 1.0
    assert float(dev_e.sum()) > 1.0
    # The content is in different places: the centered frame should have
    # more deviation in the center region than the edge-positioned one.
    center_mask = torch.zeros_like(dev_c, dtype=torch.bool)
    center_mask[:, 10:22, 10:22] = True
    assert dev_c[center_mask].sum() > dev_e[center_mask].sum()


def test_rotation_preserves_total_content(world: ProceduralWorld) -> None:
    """Rotating the token by 90° keeps total absolute deviation comparable
    (up to bilinear-interpolation blur)."""
    f0 = world.render(_state(theta=0.0, contrast=1.0))
    f90 = world.render(_state(theta=90.0, contrast=1.0))
    dev0 = float((f0 - TARGET_MEAN).abs().sum())
    dev90 = float((f90 - TARGET_MEAN).abs().sum())
    # Within ±30 % — bilinear interpolation smears some energy out.
    ratio = dev90 / max(dev0, 1e-8)
    assert 0.7 < ratio < 1.3, (
        f"rotation-90 total abs deviation ratio {ratio:.4f} — excessive "
        f"content loss (baseline={dev0:.2f}, rotated={dev90:.2f})"
    )


def test_frame_mean_near_half(world: ProceduralWorld) -> None:
    """Background-dominated frames hover near 0.5 mean luminance."""
    frames, _ = world.trajectory(0, n_steps=100)
    for i in range(frames.shape[0]):
        assert abs(float(frames[i].mean()) - TARGET_MEAN) < 0.15, (
            f"frame {i} mean {float(frames[i].mean()):.4f} — way off 0.5"
        )
