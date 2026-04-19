"""Load-bearing test for v4 D.18 patch — energy penalty is pathway-agnostic.

The v4 plan patch explicitly requires the circuit-wide energy cost to apply
uniformly to feedforward, recurrent, and feedback synapses. An earlier
draft made the L2 current penalty feedback-specific, which would have
biased the model against the very mechanism under study.

This test constructs three weight tensors representing feedforward,
recurrent, and feedback connections (named only by test convention — the
energy module must not distinguish them), all with identical values and
shape, applies the same `pre_activity`, and confirms that the returned
shrinkage tensors are bit-identical.

If this test ever fails, the energy module has become pathway-specific and
the v4 patch is broken — NO-GO.
"""

from __future__ import annotations

import torch

from src.v2_model.energy import EnergyPenalty


def test_shrinkage_is_identical_across_labelled_pathways() -> None:
    """Three identically-valued weight tensors labelled ff/rec/fb → identical ΔW."""
    torch.manual_seed(0)
    n_post, n_pre = 8, 6
    base = torch.randn(n_post, n_pre)
    pre = torch.randn(4, n_pre)

    # Same data, three "semantic" pathway labels — the module must not
    # treat them differently.
    weights_feedforward = base.clone()
    weights_recurrent = base.clone()
    weights_feedback = base.clone()

    energy = EnergyPenalty(alpha=1e-3, beta=1e-4)

    dw_ff = energy.current_weight_shrinkage(weights_feedforward, pre)
    dw_rec = energy.current_weight_shrinkage(weights_recurrent, pre)
    dw_fb = energy.current_weight_shrinkage(weights_feedback, pre)

    torch.testing.assert_close(dw_ff, dw_rec, atol=0.0, rtol=0.0)
    torch.testing.assert_close(dw_rec, dw_fb, atol=0.0, rtol=0.0)


def test_api_takes_no_pathway_argument() -> None:
    """`current_weight_shrinkage`'s signature must not have a pathway kwarg.

    Fails-to-call would surface a ``TypeError`` here; the absence of the
    kwarg from the signature inspection is what we assert on.
    """
    import inspect

    energy = EnergyPenalty(alpha=1e-3, beta=1e-4)
    sig = inspect.signature(energy.current_weight_shrinkage)
    params = set(sig.parameters.keys())
    # Required sig: (weights, pre_activity, mask). Anything matching
    # /pathway|kind|class|feedback|recurrent|feedforward/i is a regression.
    forbidden_substrings = (
        "pathway", "kind", "class",
        "feedback", "recurrent", "feedforward",
    )
    for name in params:
        for s in forbidden_substrings:
            assert s not in name.lower(), (
                f"current_weight_shrinkage parameter `{name}` suggests "
                f"pathway-specific behaviour — violates v4 D.18 patch."
            )


def test_rate_penalty_pathway_agnostic() -> None:
    """Rate penalty is also pathway-agnostic: no label in its signature."""
    import inspect

    energy = EnergyPenalty(alpha=1e-3, beta=1e-4)
    sig = inspect.signature(energy.rate_penalty_delta_drive)
    params = set(sig.parameters.keys())
    forbidden_substrings = ("pathway", "kind", "class",
                            "feedback", "recurrent", "feedforward")
    for name in params:
        for s in forbidden_substrings:
            assert s not in name.lower()


def test_identical_stats_give_identical_shrinkage_even_at_different_indices() -> None:
    """Permuting the presynaptic index should permute the shrinkage identically
    — there is no positional bias applied by the module."""
    torch.manual_seed(1)
    n_pre = 6
    pre = torch.randn(4, n_pre)
    weights = torch.randn(8, n_pre)
    energy = EnergyPenalty(alpha=0.0, beta=1e-3)

    perm = torch.randperm(n_pre)
    pre_perm = pre[:, perm]
    w_perm = weights[:, perm]

    dw_base = energy.current_weight_shrinkage(weights, pre)
    dw_perm = energy.current_weight_shrinkage(w_perm, pre_perm)

    # Permuting the pre-index should permute columns of ΔW identically.
    torch.testing.assert_close(dw_base[:, perm], dw_perm, atol=0.0, rtol=0.0)
