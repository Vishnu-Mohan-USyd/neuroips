"""Unit tests for SpikingNetworkState (src/spiking/state.py)."""

from __future__ import annotations

import pytest
import torch

from src.spiking.state import SpikingNetworkState, initial_spiking_state


# ----------------------------------------------------------------------------
# Structure
# ----------------------------------------------------------------------------

class TestSpikingNetworkStateStructure:
    def test_has_19_fields(self):
        """Plan lists 19 distinct fields across 7 groups."""
        assert len(SpikingNetworkState._fields) == 19

    def test_field_names_match_plan(self):
        """Exact field list from plan lines 100-128."""
        expected = (
            # L4 (ALIF)
            "v_l4", "z_l4", "x_l4", "adapt_l4",
            # PV (rate)
            "r_pv",
            # L2/3 (RLIF)
            "v_l23", "z_l23", "x_l23",
            # SOM (LIF)
            "v_som", "z_som", "x_som",
            # VIP (LIF)
            "v_vip", "z_vip", "x_vip",
            # V2 (LSNN)
            "v_v2", "z_v2", "x_v2", "b_v2",
            # Shared
            "deep_template",
        )
        assert SpikingNetworkState._fields == expected

    def test_has_all_three_states_per_population(self):
        """Every spiking population has v_, z_, x_ fields."""
        fields = set(SpikingNetworkState._fields)
        for pop in ("l4", "l23", "som", "vip", "v2"):
            assert f"v_{pop}" in fields, f"missing v_{pop}"
            assert f"z_{pop}" in fields, f"missing z_{pop}"
            assert f"x_{pop}" in fields, f"missing x_{pop}"

    def test_pv_is_rate_based(self):
        """PV keeps its rate field (no membrane/spike)."""
        fields = set(SpikingNetworkState._fields)
        assert "r_pv" in fields
        assert "v_pv" not in fields
        assert "z_pv" not in fields

    def test_adaptive_populations_have_adaptation(self):
        """L4 (SSA) and V2 (LSNN) have adaptation state."""
        fields = set(SpikingNetworkState._fields)
        assert "adapt_l4" in fields  # L4 SSA (tau_a=200)
        assert "b_v2" in fields       # V2 LSNN adaptive threshold


# ----------------------------------------------------------------------------
# initial_spiking_state — defaults
# ----------------------------------------------------------------------------

class TestInitialSpikingStateDefaults:
    def test_returns_spiking_network_state(self):
        s = initial_spiking_state(batch_size=4)
        assert isinstance(s, SpikingNetworkState)

    def test_batch_size_respected(self):
        for B in [1, 4, 32, 128]:
            s = initial_spiking_state(batch_size=B)
            assert s.v_l4.shape[0] == B
            assert s.r_pv.shape[0] == B
            assert s.v_v2.shape[0] == B

    def test_default_shapes(self):
        """Default n_orientations=36, v2_hidden_dim=80."""
        s = initial_spiking_state(batch_size=8)

        # V1 populations: [B, 36]
        for field in ("v_l4", "z_l4", "x_l4", "adapt_l4",
                      "v_l23", "z_l23", "x_l23",
                      "v_som", "z_som", "x_som",
                      "v_vip", "z_vip", "x_vip",
                      "deep_template"):
            t = getattr(s, field)
            assert t.shape == (8, 36), f"{field} has shape {t.shape}, expected (8, 36)"

        # PV: [B, 1]
        assert s.r_pv.shape == (8, 1)

        # V2: [B, 80]
        for field in ("v_v2", "z_v2", "x_v2", "b_v2"):
            t = getattr(s, field)
            assert t.shape == (8, 80), f"{field} has shape {t.shape}, expected (8, 80)"

    def test_all_zero_initialized(self):
        s = initial_spiking_state(batch_size=4)
        for field in SpikingNetworkState._fields:
            t = getattr(s, field)
            assert torch.all(t == 0.0), f"{field} is not zero-initialized"

    def test_default_dtype_is_float32(self):
        s = initial_spiking_state(batch_size=4)
        for field in SpikingNetworkState._fields:
            t = getattr(s, field)
            assert t.dtype == torch.float32


# ----------------------------------------------------------------------------
# initial_spiking_state — overrides
# ----------------------------------------------------------------------------

class TestInitialSpikingStateOverrides:
    def test_custom_n_orientations(self):
        s = initial_spiking_state(batch_size=2, n_orientations=72)
        assert s.v_l4.shape == (2, 72)
        assert s.v_l23.shape == (2, 72)
        assert s.r_pv.shape == (2, 1)  # PV stays scalar regardless
        assert s.v_v2.shape == (2, 80)  # V2 stays at default

    def test_custom_v2_hidden_dim(self):
        s = initial_spiking_state(batch_size=2, v2_hidden_dim=120)
        assert s.v_v2.shape == (2, 120)
        assert s.z_v2.shape == (2, 120)
        assert s.x_v2.shape == (2, 120)
        assert s.b_v2.shape == (2, 120)
        assert s.v_l4.shape == (2, 36)  # V1 stays at default

    def test_custom_dtype(self):
        s = initial_spiking_state(batch_size=2, dtype=torch.float64)
        for field in SpikingNetworkState._fields:
            t = getattr(s, field)
            assert t.dtype == torch.float64

    def test_device_placement_cpu(self):
        s = initial_spiking_state(batch_size=2, device=torch.device("cpu"))
        for field in SpikingNetworkState._fields:
            t = getattr(s, field)
            assert t.device.type == "cpu"


# ----------------------------------------------------------------------------
# NamedTuple behaviour
# ----------------------------------------------------------------------------

class TestNamedTupleBehaviour:
    def test_attribute_access(self):
        s = initial_spiking_state(batch_size=4)
        assert s.v_l4 is s[0]  # position 0
        assert s.deep_template is s[-1]  # last field

    def test_replace_returns_new_instance(self):
        """_replace is the canonical way to update a single field (immutable update)."""
        s = initial_spiking_state(batch_size=2)
        new_v_l4 = torch.ones(2, 36)
        s2 = s._replace(v_l4=new_v_l4)
        # Original unchanged
        assert torch.all(s.v_l4 == 0.0)
        # New state has updated field
        assert torch.all(s2.v_l4 == 1.0)
        # Other fields identical (by identity — NamedTuple shares refs)
        assert s2.r_pv is s.r_pv
        assert s2.deep_template is s.deep_template

    def test_can_unpack(self):
        """NamedTuple supports positional unpacking (* pattern)."""
        s = initial_spiking_state(batch_size=1)
        items = list(s)
        assert len(items) == 19
        assert all(isinstance(t, torch.Tensor) for t in items)


# ----------------------------------------------------------------------------
# Interop with rate NetworkState (sanity — different structure, same role)
# ----------------------------------------------------------------------------

class TestRateParityShapes:
    def test_l23_shape_matches_rate_model(self):
        """x_l23 has the same shape as r_l23 in the rate model."""
        from src.state import initial_state as initial_rate_state

        rate = initial_rate_state(batch_size=4, n_orientations=36)
        spk = initial_spiking_state(batch_size=4, n_orientations=36)
        assert spk.x_l23.shape == rate.r_l23.shape

    def test_pv_shape_matches_rate_model(self):
        """r_pv is unchanged between rate and spiking states."""
        from src.state import initial_state as initial_rate_state

        rate = initial_rate_state(batch_size=4)
        spk = initial_spiking_state(batch_size=4)
        assert spk.r_pv.shape == rate.r_pv.shape

    def test_deep_template_unchanged(self):
        from src.state import initial_state as initial_rate_state

        rate = initial_rate_state(batch_size=4)
        spk = initial_spiking_state(batch_size=4)
        assert spk.deep_template.shape == rate.deep_template.shape
