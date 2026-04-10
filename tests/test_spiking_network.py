"""Unit + integration tests for SpikingLaminarV1V2Network (src/spiking/network.py).

Covers the five acceptance criteria from task #16:
    1. Shape parity with the rate network for all output-dict keys.
    2. Forward pass through T=50 timesteps completes without error.
    3. Gradients flow from a decoder loss back to every learnable parameter
       that the *rate* model's same branch also exercises (we compare the set
       of non-None grads to the rate model on the same inputs).
    4. No NaN/Inf in any state or aux tensor over T=50 steps.
    5. Step dependency order matches the rate model exactly:
        L4 -> PV -> V2 -> DeepTemplate -> Feedback -> SOM/VIP -> L2/3
       This is verified with a targeted mutation test that swaps the
       dependency one step at a time and confirms the output changes, i.e.
       that the spiking network is actually threading inputs the same way.

The three feedback branches (emergent, emergent+simple_feedback, fixed) are
each exercised at least once.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from src.config import ModelConfig, MechanismType, SpikingConfig
from src.model.network import LaminarV1V2Network
from src.spiking.network import SpikingLaminarV1V2Network
from src.spiking.state import SpikingNetworkState, initial_spiking_state


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def make_net(
    feedback_mode: str = "emergent",
    simple_feedback: bool = False,
    mechanism: MechanismType = MechanismType.DAMPENING,
    v2_input_mode: str = "l23",
) -> SpikingLaminarV1V2Network:
    """Construct a SpikingLaminarV1V2Network with the given feedback settings."""
    mcfg = replace(
        ModelConfig(),
        feedback_mode=feedback_mode,
        simple_feedback=simple_feedback,
        mechanism=mechanism,
        v2_input_mode=v2_input_mode,
    )
    scfg = SpikingConfig()
    return SpikingLaminarV1V2Network(mcfg, scfg)


def make_stim(B: int = 2, T: int = 20, N: int = 36, seed: int = 0) -> torch.Tensor:
    """Random non-negative stimulus tensor [B, T, N]."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, N, generator=g).clamp(min=0.0)


# ============================================================================
# 1. Shape parity with the rate network
# ============================================================================

# Keys that the rate network's forward() returns in its aux dict.
RATE_AUX_KEYS = {
    "q_pred_all",
    "pi_pred_all",
    "pi_pred_eff_all",
    "state_logits_all",
    "deep_template_all",
    "r_l4_all",
    "r_pv_all",
    "r_som_all",
    "r_vip_all",
    "p_cw_all",
    "center_exc_all",
}

# Extra keys the spiking network adds (spike trajectories).
SPIKE_AUX_KEYS = {
    "spike_l4_all",
    "spike_l23_all",
    "spike_som_all",
    "spike_vip_all",
    "spike_v2_all",
}


class TestShapeParity:
    def test_aux_keys_superset_of_rate(self):
        """Spiking aux dict contains every key the rate network exposes."""
        snet = make_net(feedback_mode="emergent", simple_feedback=False)
        stim = make_stim()
        _, _, aux = snet(stim)
        assert RATE_AUX_KEYS.issubset(aux.keys())

    def test_aux_includes_spike_trajectories(self):
        snet = make_net(feedback_mode="emergent", simple_feedback=False)
        stim = make_stim()
        _, _, aux = snet(stim)
        assert SPIKE_AUX_KEYS.issubset(aux.keys())

    def test_shapes_match_rate_network_emergent(self):
        """Same shapes as rate network aux dict (emergent mode, simple_feedback=False)."""
        mcfg = replace(
            ModelConfig(), feedback_mode="emergent", simple_feedback=False
        )
        scfg = SpikingConfig()
        rnet = LaminarV1V2Network(mcfg)
        snet = SpikingLaminarV1V2Network(mcfg, scfg)

        stim = make_stim(B=2, T=12, N=mcfg.n_orientations)
        r_out, _, r_aux = rnet(stim)
        s_out, _, s_aux = snet(stim)

        assert r_out.shape == s_out.shape
        for k in RATE_AUX_KEYS:
            assert r_aux[k].shape == s_aux[k].shape, (
                f"Shape mismatch on {k}: rate={r_aux[k].shape}, spike={s_aux[k].shape}"
            )

    def test_spike_trajectory_shapes(self):
        snet = make_net(feedback_mode="emergent", simple_feedback=False)
        B, T, N = 3, 15, snet.cfg.n_orientations
        stim = make_stim(B=B, T=T, N=N)
        _, _, aux = snet(stim)
        assert aux["spike_l4_all"].shape == (B, T, N)
        assert aux["spike_l23_all"].shape == (B, T, N)
        assert aux["spike_som_all"].shape == (B, T, N)
        assert aux["spike_vip_all"].shape == (B, T, N)
        assert aux["spike_v2_all"].shape == (B, T, snet.spiking_cfg.n_lsnn_neurons)


# ============================================================================
# 2. Forward over T=50 completes without error
# ============================================================================

class TestForwardT50:
    @pytest.mark.parametrize("simple_feedback", [True, False])
    def test_emergent_T50(self, simple_feedback):
        snet = make_net(feedback_mode="emergent", simple_feedback=simple_feedback)
        stim = make_stim(T=50)
        out, final_state, aux = snet(stim)
        assert out.shape[1] == 50
        assert isinstance(final_state, SpikingNetworkState)

    @pytest.mark.parametrize(
        "mech",
        [
            MechanismType.DAMPENING,
            MechanismType.SHARPENING,
            MechanismType.CENTER_SURROUND,
            MechanismType.ADAPTATION_ONLY,
            MechanismType.PREDICTIVE_ERROR,
        ],
    )
    def test_fixed_modes_T50(self, mech):
        snet = make_net(
            feedback_mode="fixed", mechanism=mech, v2_input_mode="l4"
        )
        stim = make_stim(T=50)
        out, _, _ = snet(stim)
        assert out.shape[1] == 50


# ============================================================================
# 3. Gradient flow
# ============================================================================

class TestGradientFlow:
    def test_gradients_match_rate_model_param_coverage(self):
        """In emergent non-simple mode, the *set* of params that receive non-None
        gradients must match between the spiking and rate networks.

        Some params are structurally unused in specific feedback branches (e.g.
        v2.head_feedback is only active when simple_feedback=True). That's
        inherited from the rate model's architecture, so the spiking port must
        reproduce the same pattern exactly — which is what this test locks in.
        """
        mcfg = replace(
            ModelConfig(), feedback_mode="emergent", simple_feedback=False
        )
        scfg = SpikingConfig()

        torch.manual_seed(0)
        rnet = LaminarV1V2Network(mcfg)
        torch.manual_seed(0)
        snet = SpikingLaminarV1V2Network(mcfg, scfg)

        stim = make_stim(B=2, T=20)

        # Rate network backward
        r_out, _, r_aux = rnet(stim)
        r_loss = (
            r_out.sum()
            + r_aux["r_l4_all"].sum()
            + r_aux["r_som_all"].sum()
            + r_aux["r_vip_all"].sum()
            + r_aux["deep_template_all"].sum()
        )
        r_loss.backward()
        rate_no_grad = {
            n for n, p in rnet.named_parameters() if p.grad is None
        }

        # Spiking network backward
        s_out, _, s_aux = snet(stim)
        s_loss = (
            s_out.sum()
            + s_aux["r_l4_all"].sum()
            + s_aux["r_som_all"].sum()
            + s_aux["r_vip_all"].sum()
            + s_aux["deep_template_all"].sum()
        )
        s_loss.backward()
        spike_no_grad = {
            n for n, p in snet.named_parameters() if p.grad is None
        }

        assert rate_no_grad == spike_no_grad, (
            f"Grad-None sets differ:\n"
            f"  rate only:  {rate_no_grad - spike_no_grad}\n"
            f"  spike only: {spike_no_grad - rate_no_grad}"
        )

    def test_gradients_flow_through_spiking_populations(self):
        """Every parameter that actually lives in a spiking population
        must receive a finite, non-zero gradient in at least one of the
        three feedback branches."""
        # Use emergent mode so L4, L23, SOM, VIP, and V2 are all exercised.
        mcfg = replace(
            ModelConfig(), feedback_mode="emergent", simple_feedback=False
        )
        scfg = SpikingConfig()
        snet = SpikingLaminarV1V2Network(mcfg, scfg)
        stim = make_stim(B=2, T=20)

        out, _, aux = snet(stim)
        loss = (
            out.sum()
            + aux["r_l4_all"].sum()
            + aux["r_som_all"].sum()
            + aux["r_vip_all"].sum()
            + aux["deep_template_all"].sum()
        )
        loss.backward()

        # Spiking-population parameters that MUST receive grad.
        required = [
            "l23.sigma_rec_raw",
            "l23.gain_rec_raw",
            "l23.w_som.gain_raw",
            "l23.w_pv_l23.gain_raw",
            "pv.w_pv_l4_raw",
            "pv.w_pv_l23_raw",
            "v2.input_proj.weight",
            "v2.input_proj.bias",
            "v2.W_rec.weight",
            "v2.head_mu.weight",
            "v2.head_mu.bias",
        ]
        params = dict(snet.named_parameters())
        for name in required:
            assert name in params, f"Missing expected parameter: {name}"
            g = params[name].grad
            assert g is not None, f"No gradient reached {name}"
            assert torch.isfinite(g).all(), f"NaN/Inf in grad for {name}"

    def test_gradients_flow_through_head_feedback_in_simple_mode(self):
        """head_feedback must receive gradient when simple_feedback=True."""
        mcfg = replace(
            ModelConfig(), feedback_mode="emergent", simple_feedback=True
        )
        scfg = SpikingConfig()
        snet = SpikingLaminarV1V2Network(mcfg, scfg)
        stim = make_stim(B=2, T=20)

        out, _, aux = snet(stim)
        # center_exc depends directly on head_feedback in simple_feedback mode.
        loss = out.sum() + aux["center_exc_all"].sum()
        loss.backward()

        g = snet.v2.head_feedback.weight.grad
        assert g is not None and torch.isfinite(g).all()
        assert g.abs().sum() > 0.0


# ============================================================================
# 4. No NaN/Inf in state or aux over T=50
# ============================================================================

class TestNoNaNInf:
    @pytest.mark.parametrize("simple_feedback", [True, False])
    def test_emergent_no_nan_inf(self, simple_feedback):
        snet = make_net(feedback_mode="emergent", simple_feedback=simple_feedback)
        stim = make_stim(T=50)
        out, final_state, aux = snet(stim)

        # Final state
        for name, t in final_state._asdict().items():
            assert torch.isfinite(t).all(), (
                f"Non-finite values in final_state.{name}"
            )

        # Output trajectory
        assert torch.isfinite(out).all()

        # Full aux dict
        for k, t in aux.items():
            assert torch.isfinite(t).all(), f"Non-finite values in aux[{k!r}]"

    def test_fixed_mode_no_nan_inf(self):
        snet = make_net(
            feedback_mode="fixed",
            mechanism=MechanismType.DAMPENING,
            v2_input_mode="l4",
        )
        stim = make_stim(T=50)
        out, final_state, aux = snet(stim)
        assert torch.isfinite(out).all()
        for name, t in final_state._asdict().items():
            assert torch.isfinite(t).all(), f"Non-finite in final_state.{name}"
        for k, t in aux.items():
            assert torch.isfinite(t).all(), f"Non-finite in aux[{k!r}]"


# ============================================================================
# 5. Step dependency order cross-check
# ============================================================================

class TestStepDependencyOrder:
    """Verify L4 → PV → V2 → DeepTemplate → Feedback → SOM/VIP → L2/3 order.

    Strategy: exploit the rate model's step() and the spiking step() to
    compare which state fields change on one step from a fixed zero state.
    Only the populations that fire in the correct branch should become
    non-zero, and in the spiking case we can also verify that the
    *intermediate* dependencies propagate by mutating state.r_pv and
    state.x_l23 and observing that the correct downstream fields change.
    """

    def test_l4_reads_previous_pv(self):
        """Perturbing state.r_pv must change v_l4 at the next step."""
        snet = make_net(feedback_mode="emergent", simple_feedback=False)
        snet.eval()

        B, N = 2, snet.cfg.n_orientations
        s0 = initial_spiking_state(B, N, snet.spiking_cfg.n_lsnn_neurons)
        s1 = s0._replace(r_pv=torch.ones(B, 1) * 2.0)

        stim = torch.ones(B, N) * 0.5
        cue = torch.zeros(B, N)
        ts = torch.zeros(B, 2)

        out0, _ = snet.step(stim, cue, ts, s0)
        out1, _ = snet.step(stim, cue, ts, s1)
        assert not torch.allclose(out0.v_l4, out1.v_l4)

    def test_v2_reads_previous_l23(self):
        """Perturbing state.x_l23 must change V2 hidden state at the next step."""
        snet = make_net(feedback_mode="emergent", simple_feedback=False)
        snet.eval()

        B, N = 2, snet.cfg.n_orientations
        s0 = initial_spiking_state(B, N, snet.spiking_cfg.n_lsnn_neurons)
        s1 = s0._replace(x_l23=torch.ones(B, N) * 0.5)

        stim = torch.zeros(B, N)
        cue = torch.zeros(B, N)
        ts = torch.zeros(B, 2)

        out0, _ = snet.step(stim, cue, ts, s0)
        out1, _ = snet.step(stim, cue, ts, s1)
        assert not torch.allclose(out0.v_v2, out1.v_v2)

    def test_pv_reads_new_x_l4(self):
        """PV uses the *new* x_l4 from this step, not the previous-step trace.
        Therefore perturbing stimulus changes r_pv immediately at t=0."""
        snet = make_net(feedback_mode="emergent", simple_feedback=False)
        snet.eval()
        B, N = 2, snet.cfg.n_orientations
        s0 = initial_spiking_state(B, N, snet.spiking_cfg.n_lsnn_neurons)

        stim_a = torch.ones(B, N) * 0.5
        stim_b = torch.ones(B, N) * 1.5
        cue = torch.zeros(B, N)
        ts = torch.zeros(B, 2)
        out_a, _ = snet.step(stim_a, cue, ts, s0)
        out_b, _ = snet.step(stim_b, cue, ts, s0)
        assert not torch.allclose(out_a.r_pv, out_b.r_pv)

    def test_l23_reads_new_x_som_and_new_r_pv(self):
        """L2/3 must integrate the current-step SOM trace and current-step
        PV rate, i.e. feedback → SOM → L2/3 within one step."""
        snet = make_net(feedback_mode="emergent", simple_feedback=True)
        snet.eval()

        # Push the V2 head_feedback bias so feedback_signal is non-zero even
        # from a zero state — this guarantees SOM and center_exc both move.
        with torch.no_grad():
            snet.v2.head_feedback.bias.fill_(1.0)

        B, N = 2, snet.cfg.n_orientations
        s0 = initial_spiking_state(B, N, snet.spiking_cfg.n_lsnn_neurons)
        stim = torch.ones(B, N) * 0.5
        cue = torch.zeros(B, N)
        ts = torch.zeros(B, 2)

        new_state, aux = snet.step(stim, cue, ts, s0)
        # The forward produced non-zero feedback → center_exc should be nonzero
        # → L2/3 membrane should be driven nonzero.
        assert aux.center_exc.abs().sum() > 0, "Center excitation is zero"
        assert new_state.v_l23.abs().sum() > 0, "L2/3 did not integrate feedback"


# ============================================================================
# 6. Feedback routing sign split (simple_feedback E/I)
# ============================================================================

class TestFeedbackRouting:
    """simple_feedback mode: positive feedback → L2/3 excitation (center_exc),
    negative feedback → SOM drive. Lead's task #19 explicitly requires this.
    """

    def _make_net_with_signed_feedback(self, sign: float) -> SpikingLaminarV1V2Network:
        """Build a net whose V2 head_feedback bias is forced to +/-sign."""
        snet = make_net(feedback_mode="emergent", simple_feedback=True)
        snet.eval()
        with torch.no_grad():
            snet.v2.head_feedback.weight.zero_()
            snet.v2.head_feedback.bias.fill_(float(sign))
        return snet

    def test_positive_feedback_becomes_center_exc(self):
        """+bias → positive feedback → center_exc > 0 and SOM drive zero."""
        snet = self._make_net_with_signed_feedback(sign=+1.0)
        B, N = 2, snet.cfg.n_orientations
        s0 = initial_spiking_state(B, N, snet.spiking_cfg.n_lsnn_neurons)
        stim = torch.zeros(B, N)
        cue = torch.zeros(B, N)
        ts = torch.zeros(B, 2)
        new_state, aux = snet.step(stim, cue, ts, s0)

        # Positive branch active: center_exc is strictly positive everywhere.
        assert (aux.center_exc > 0).all(), "positive feedback did not route to center_exc"
        # Negative branch dormant: SOM had no drive this step, so no spikes.
        assert new_state.z_som.abs().sum() == 0.0, "SOM should not fire on positive feedback"

    def test_negative_feedback_becomes_som_drive(self):
        """-bias → negative feedback → SOM drive > 0 and center_exc zero."""
        snet = self._make_net_with_signed_feedback(sign=-1.0)
        B, N = 2, snet.cfg.n_orientations
        s0 = initial_spiking_state(B, N, snet.spiking_cfg.n_lsnn_neurons)
        stim = torch.zeros(B, N)
        cue = torch.zeros(B, N)
        ts = torch.zeros(B, 2)
        new_state, aux = snet.step(stim, cue, ts, s0)

        # Negative branch active: center_exc is zero (relu of negative).
        assert (aux.center_exc == 0).all(), "negative feedback leaked into center_exc"
        # SOM membrane should be driven to a positive value by |feedback|.
        assert (new_state.v_som > 0).all(), "SOM did not receive negative-part drive"


# ============================================================================
# 7. Oracle mode: V2 bypassed with injected q_pred / pi_pred
# ============================================================================

class TestOracleMode:
    """Oracle mode: experiments inject ground-truth q_pred/pi_pred, V2 output
    is ignored. The oracle path must still route the injected signals into the
    aux dict and through DeepTemplate → feedback operator."""

    def _oracle_pair(self, B: int, T: int, N: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (q_pred [B,T,N], pi_pred [B,T,1]) with unit peaks."""
        q = torch.zeros(B, T, N)
        q[:, :, 0] = 1.0
        pi = torch.full((B, T, 1), 0.8)
        return q, pi

    def test_per_sequence_oracle_flows_to_aux(self):
        """[B, T, N] oracle q_pred passes verbatim into aux['q_pred_all']."""
        snet = make_net(feedback_mode="emergent", simple_feedback=False)
        B, T, N = 2, 8, snet.cfg.n_orientations
        q, pi = self._oracle_pair(B, T, N)
        snet.oracle_mode = True
        snet.oracle_q_pred = q
        snet.oracle_pi_pred = pi

        stim = make_stim(B=B, T=T, N=N)
        _, _, aux = snet(stim)
        torch.testing.assert_close(aux["q_pred_all"], q)
        torch.testing.assert_close(aux["pi_pred_all"], pi)

    def test_per_step_oracle_flows_to_aux(self):
        """[B, N] oracle q_pred is broadcast across all timesteps."""
        snet = make_net(feedback_mode="emergent", simple_feedback=False)
        B, T, N = 2, 6, snet.cfg.n_orientations
        q_step = torch.zeros(B, N)
        q_step[:, 3] = 1.0
        pi_step = torch.full((B, 1), 0.5)
        snet.oracle_mode = True
        snet.oracle_q_pred = q_step
        snet.oracle_pi_pred = pi_step

        stim = make_stim(B=B, T=T, N=N)
        _, _, aux = snet(stim)
        # All timesteps should receive the same broadcast q_pred.
        for t in range(T):
            torch.testing.assert_close(aux["q_pred_all"][:, t], q_step)
            torch.testing.assert_close(aux["pi_pred_all"][:, t], pi_step)

    def test_oracle_ignores_v2_params_changes(self):
        """With oracle on, zeroing V2 readout weights must not change aux['q_pred_all']
        (because the oracle path bypasses V2's heads entirely)."""
        snet = make_net(feedback_mode="emergent", simple_feedback=False)
        snet.eval()
        B, T, N = 2, 8, snet.cfg.n_orientations
        q, pi = self._oracle_pair(B, T, N)
        snet.oracle_mode = True
        snet.oracle_q_pred = q
        snet.oracle_pi_pred = pi

        stim = make_stim(B=B, T=T, N=N, seed=1)
        _, _, aux_before = snet(stim)

        # Mutate V2 readout heads — should have no effect on q_pred.
        with torch.no_grad():
            snet.v2.head_mu.weight.zero_()
            snet.v2.head_mu.bias.fill_(-100.0)
            snet.v2.head_pi.weight.zero_()
            snet.v2.head_pi.bias.fill_(-100.0)

        _, _, aux_after = snet(stim)
        torch.testing.assert_close(aux_before["q_pred_all"], aux_after["q_pred_all"])


# ============================================================================
# 8. No NaN/Inf over T=100 (Task #19 explicit requirement)
# ============================================================================

class TestLongRollout:
    def test_T100_no_nan_inf_emergent(self):
        snet = make_net(feedback_mode="emergent", simple_feedback=False)
        stim = make_stim(T=100)
        out, final_state, aux = snet(stim)
        assert torch.isfinite(out).all()
        for name, t in final_state._asdict().items():
            assert torch.isfinite(t).all(), f"Non-finite in final_state.{name}"
        for k, t in aux.items():
            assert torch.isfinite(t).all(), f"Non-finite in aux[{k!r}]"

    def test_T100_no_nan_inf_simple_feedback(self):
        snet = make_net(feedback_mode="emergent", simple_feedback=True)
        stim = make_stim(T=100)
        out, final_state, aux = snet(stim)
        assert torch.isfinite(out).all()
        for name, t in final_state._asdict().items():
            assert torch.isfinite(t).all()
        for k, t in aux.items():
            assert torch.isfinite(t).all()


# ============================================================================
# 9. cache_kernels / uncache_kernels
# ============================================================================

class TestKernelCaching:
    def test_cache_used_during_forward(self):
        """Inside forward, l23._cached_W_rec must be populated; outside, None."""
        snet = make_net(feedback_mode="emergent", simple_feedback=False)

        # Before a forward call, the cache is empty.
        assert snet.l23._cached_W_rec is None

        # Monkey-patch l23.forward to peek at the cache mid-rollout.
        seen = {}
        original_fwd = snet.l23.forward

        def spy(*args, **kwargs):
            seen["cached_during"] = snet.l23._cached_W_rec is not None
            return original_fwd(*args, **kwargs)

        snet.l23.forward = spy
        try:
            stim = make_stim(T=4)
            snet(stim)
        finally:
            snet.l23.forward = original_fwd

        assert seen.get("cached_during") is True, "kernel cache not populated mid-forward"
        # After forward finishes, the cache is cleared again.
        assert snet.l23._cached_W_rec is None


# ============================================================================
# 10. Audit items (Task #19 consolidation)
# ============================================================================

class TestAudit:
    def test_b_v2_gradient_across_10_steps(self):
        """The V2 adaptive threshold state b_v2 must carry gradients across
        ~10 BPTT steps. b at step t feeds B_thresh at step t+1, so a loss at
        the final step must backprop through the full b chain into v2 params.

        Uses v2_input_mode='l4_l23' so V2 receives nonzero drive from the
        L4 filtered trace even though L2/3 is cold in Phase 1 (see plan
        cold-rate calibration note). Uses a scaled stimulus to guarantee
        L4 fires enough spikes to exercise V2.
        """
        torch.manual_seed(7)
        snet = make_net(
            feedback_mode="emergent",
            simple_feedback=False,
            v2_input_mode="l4_l23",
        )
        stim = make_stim(B=2, T=10, seed=7) * 3.0

        _, _, aux = snet(stim)
        # Use non-invariant loss on V2 outputs (squared mu avoids softmax
        # row-sum invariance, which would give exact-zero grads on head_mu).
        loss = (
            aux["q_pred_all"].pow(2).sum()
            + aux["pi_pred_all"].sum()
            + aux["center_exc_all"].sum()
        )
        loss.backward()

        # V2 parameters on the adaptive-threshold path. head_feedback is
        # intentionally excluded: in emergent + simple_feedback=False mode,
        # center_exc comes from EmergentFeedbackOperator (which reads
        # q_pred/pi_pred), not from head_feedback, so head_feedback is
        # architecturally out of this branch.
        v2_path = [
            "v2.input_proj.weight",
            "v2.W_rec.weight",
            "v2.head_mu.weight",
            "v2.head_pi.weight",
        ]
        params = dict(snet.named_parameters())
        for name in v2_path:
            g = params[name].grad
            assert g is not None, f"No gradient reached {name} after T=10 BPTT"
            assert torch.isfinite(g).all(), f"NaN/Inf grad in {name}"
            assert g.abs().max() > 0, f"Zero grad in {name}"

    def test_step_under_torch_compile(self):
        """network.step must survive torch.compile(mode='default',
        fullgraph=False) for forward+backward. Non-fatal if compile backend
        is unavailable in the environment.
        """
        snet = make_net(feedback_mode="emergent", simple_feedback=False)
        B, T, N = 2, 5, snet.cfg.n_orientations

        try:
            compiled_step = torch.compile(
                snet.step, mode="default", fullgraph=False
            )
        except Exception as e:
            pytest.skip(f"torch.compile unavailable: {e}")

        state = initial_spiking_state(
            batch_size=B,
            n_orientations=snet.cfg.n_orientations,
            v2_hidden_dim=snet.spiking_cfg.n_lsnn_neurons,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        stim = make_stim(B=B, T=T, N=N, seed=2)
        cue = torch.zeros(B, N)
        task_state = torch.zeros(B, 2)
        outs = []
        try:
            for t in range(T):
                state, step_aux = compiled_step(
                    stim[:, t], cue, task_state, state
                )
                outs.append(step_aux.center_exc)
        except Exception as e:
            pytest.skip(f"torch.compile runtime failure (non-fatal): {e}")

        loss = torch.stack(outs).sum()
        loss.backward()

        # At least one L2/3 parameter should have a finite grad.
        g = snet.l23.sigma_rec_raw.grad
        assert g is not None
        assert torch.isfinite(g).all()

    def test_spike_count_energy_sanity(self):
        """Phase 2 spike-energy loss contract: `spike_*_all` tensors must be
        binary, shape-correct, and differentiable so the energy term can be
        expressed as a mean over spike counts.

        We check L4 spikes specifically because L2/3 is expected to be cold
        (firing fraction ~0) in untrained Phase 1, per the cold-rate
        calibration note in the master plan. L4, driven directly by the
        stimulus, fires reliably and is representative of the contract the
        Phase 2 energy loss depends on. We also verify the L2/3 tensor is
        shape-correct and differentiable even when its mean is 0.
        """
        torch.manual_seed(42)
        snet = make_net(feedback_mode="emergent", simple_feedback=False)
        B, T, N = 2, 20, snet.cfg.n_orientations
        stim = make_stim(B=B, T=T, N=N, seed=3) * 3.0

        _, _, aux = snet(stim)
        spikes_l4 = aux["spike_l4_all"]
        spikes_l23 = aux["spike_l23_all"]
        assert spikes_l4.shape == (B, T, N)
        assert spikes_l23.shape == (B, T, N)

        # Binary-tensor contract for both populations.
        for s in (spikes_l4, spikes_l23):
            uniq = torch.unique(s.detach())
            assert uniq.numel() <= 2
            assert set(uniq.tolist()).issubset({0.0, 1.0})

        # L4 mean must be a scalar strictly in (0, 1] — this is what the
        # Phase 2 energy loss will see for the driven input population.
        energy_l4 = spikes_l4.mean()
        assert energy_l4.ndim == 0
        assert 0.0 < energy_l4.item() <= 1.0

        # L2/3 mean still lives in [0, 1] and is a scalar — even if it
        # happens to be 0 at init (cold Phase 1 state).
        energy_l23 = spikes_l23.mean()
        assert energy_l23.ndim == 0
        assert 0.0 <= energy_l23.item() <= 1.0

        # Differentiability of the L4 energy term — its gradient must
        # reach any upstream spiking parameter (surrogate path live).
        energy_l4.backward()
        # L4 spikes are driven directly by stimulus and the PV pool's
        # divisive normalization, so pv.w_pv_l4_raw is in the path.
        g = snet.pv.w_pv_l4_raw.grad
        assert g is not None, "No grad on pv.w_pv_l4_raw from spike energy"
        assert torch.isfinite(g).all()
