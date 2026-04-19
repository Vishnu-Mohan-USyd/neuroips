"""Smoke tests for the Phase-1 v2_model scaffold.

Scope:
  - All v2_model submodules import cleanly.
  - Dale primitives re-exported from src.utils match the originals.
  - `initial_state` populates a NetworkStateV2 with expected shapes + dtype.
  - Scaffold-only modules (bridge, freeze_manifest) raise NotImplementedError
    on their public entry points — i.e. they advertise unfinished-ness.

No layer/plasticity/network logic is exercised here; those tests land as
Tasks #11+ land their modules.
"""

from __future__ import annotations

import importlib

import pytest
import torch

import src.utils as src_utils
import src.v2_model as v2
from src.v2_model.state import NetworkStateV2, initial_state


# ---------------------------------------------------------------------------
# Imports + re-exports
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "module_name",
    [
        "src.v2_model",
        "src.v2_model.config",
        "src.v2_model.state",
        "src.v2_model.utils",
        "src.v2_model.bridge",
        "src.v2_model.freeze_manifest",
    ],
)
def test_module_imports(module_name: str) -> None:
    """Each scaffold module must import without side effects."""
    mod = importlib.import_module(module_name)
    assert mod is not None


def test_utils_reexports_dale_primitives() -> None:
    """`src.v2_model.utils` must expose the same objects as `src.utils`."""
    from src.v2_model.utils import ExcitatoryLinear, InhibitoryGain, circular_distance

    assert ExcitatoryLinear is src_utils.ExcitatoryLinear
    assert InhibitoryGain is src_utils.InhibitoryGain
    assert circular_distance is src_utils.circular_distance


def test_package_exports_top_level_names() -> None:
    """`src.v2_model.__all__` exposes the dataclasses needed by tests + scripts."""
    expected = {
        "ArchitectureConfig",
        "ConnectivityConfig",
        "EnergyConfig",
        "FreezeManifest",
        "ModelConfig",
        "NetworkStateV2",
        "PhaseFreezeSpec",
        "PlasticityConfig",
        "RegimeConfig",
        "TimeConstantsConfig",
        "initial_state",
    }
    assert expected.issubset(set(v2.__all__))
    for name in expected:
        assert hasattr(v2, name), f"src.v2_model missing {name}"


# ---------------------------------------------------------------------------
# NetworkStateV2 shapes + dtype
# ---------------------------------------------------------------------------

def test_initial_state_returns_namedtuple(cfg, batch_size) -> None:
    state = initial_state(cfg, batch_size=batch_size)
    assert isinstance(state, NetworkStateV2)


def test_initial_state_rate_shapes(cfg, batch_size) -> None:
    """All rate vectors match the architecture-table population sizes."""
    state = initial_state(cfg, batch_size=batch_size)
    a = cfg.arch

    assert state.r_l4.shape == (batch_size, a.n_l4_e)
    assert state.r_l23.shape == (batch_size, a.n_l23_e)
    assert state.r_pv.shape == (batch_size, a.n_l23_pv)
    assert state.r_som.shape == (batch_size, a.n_l23_som)
    assert state.r_h.shape == (batch_size, a.n_h_e)
    assert state.h_pv.shape == (batch_size, a.n_h_pv)
    assert state.m.shape == (batch_size, a.n_c)


def test_initial_state_rates_zeroed(cfg, batch_size) -> None:
    """Every rate vector is zero at initialisation."""
    state = initial_state(cfg, batch_size=batch_size)
    for name in ("r_l4", "r_l23", "r_pv", "r_som", "r_h", "h_pv", "m"):
        t = getattr(state, name)
        assert torch.all(t == 0), f"{name} not zero-initialised"


def test_initial_state_regime_posterior_uniform(cfg, batch_size) -> None:
    """Regime posterior is uniform 1/n_regimes at init (no prior bias)."""
    state = initial_state(cfg, batch_size=batch_size)
    n_reg = cfg.regime.n_regimes
    assert state.regime_posterior.shape == (batch_size, n_reg)
    torch.testing.assert_close(
        state.regime_posterior.sum(dim=-1),
        torch.ones(batch_size),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        state.regime_posterior,
        torch.full((batch_size, n_reg), 1.0 / n_reg),
        atol=1e-6,
        rtol=0,
    )


def test_initial_state_traces_empty(cfg, batch_size) -> None:
    """Plasticity trace dicts start empty — plasticity modules fill them lazily."""
    state = initial_state(cfg, batch_size=batch_size)
    assert state.pre_traces == {}
    assert state.post_traces == {}


def test_initial_state_default_dtype(cfg, batch_size) -> None:
    """All rate tensors default to float32."""
    state = initial_state(cfg, batch_size=batch_size)
    for name in ("r_l4", "r_l23", "r_pv", "r_som", "r_h", "h_pv", "m",
                 "regime_posterior"):
        t = getattr(state, name)
        assert t.dtype == torch.float32, f"{name} dtype {t.dtype} != float32"


def test_initial_state_respects_explicit_device(cfg, batch_size, device) -> None:
    """Explicit `device=` argument must override cfg.device."""
    state = initial_state(cfg, batch_size=batch_size, device=device)
    assert state.r_l4.device == device


# ---------------------------------------------------------------------------
# Scaffold modules advertise unfinished state
# ---------------------------------------------------------------------------

def test_bridge_extract_activations_implemented() -> None:
    """`extract_activations` landed with Task #29; see tests/v2/test_network_bridge_extraction.py
    for full shape/determinism contract. Smoke-test here: wrong-rank stim is
    rejected (i.e. the function is no longer a pure scaffold)."""
    from src.v2_model.bridge import extract_activations
    with pytest.raises(ValueError, match=r"stim must be \[B, T, 1, H, W\]"):
        extract_activations(net=None, stim=torch.zeros(1, 1))


def test_bridge_extract_readout_data_not_implemented() -> None:
    from src.v2_model.bridge import extract_readout_data
    with pytest.raises(NotImplementedError):
        extract_readout_data(activations={}, readout_fn=lambda x: x)


def test_freeze_manifest_load_not_implemented() -> None:
    """`load_manifest` is a scaffold — must raise NotImplementedError."""
    from src.v2_model.freeze_manifest import load_manifest
    with pytest.raises(NotImplementedError):
        load_manifest("nonexistent.yaml")


def test_freeze_manifest_assert_boundary_not_implemented() -> None:
    from src.v2_model.freeze_manifest import FreezeManifest, assert_boundary
    with pytest.raises(NotImplementedError):
        assert_boundary(net=None, manifest=FreezeManifest(), phase_name="phase_2")


def test_freeze_manifest_dataclasses_instantiate(cfg) -> None:
    """Dataclasses are real — only the logic-functions are stubs."""
    from src.v2_model.freeze_manifest import FreezeManifest, PhaseFreezeSpec

    spec = PhaseFreezeSpec(
        phase_name="phase_2", plastic=["W_l23_rec"], frozen=["W_ff"]
    )
    manifest = FreezeManifest(phases={"phase_2": spec})
    assert manifest.phases["phase_2"].plastic == ["W_l23_rec"]
    assert manifest.phases["phase_2"].frozen == ["W_ff"]
