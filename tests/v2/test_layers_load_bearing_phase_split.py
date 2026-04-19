"""Phase-2 plastic manifest / Phase-3 frozen manifest — per-population.

All weights in ``layers.py`` are generic (plastic in Phase 2, frozen in
both Phase 3 variants). The task-vs-generic split for context memory
lives in ``context_memory.py``; here we only verify the binary manifest.
"""

from __future__ import annotations

from typing import Callable

import pytest

from src.v2_model.layers import HE, HPV, L23E, L23PV, L23SOM


def _l23e_factory() -> L23E:
    return L23E(
        n_units=8, n_l4_e=6, n_pv=3, n_som=4, n_h_e=5,
        tau_ms=20.0, dt_ms=5.0, seed=0,
    )


def _l23pv_factory() -> L23PV:
    return L23PV(n_units=3, n_l23_e=8, tau_ms=10.0, dt_ms=5.0, seed=0)


def _l23som_factory() -> L23SOM:
    return L23SOM(
        n_units=4, n_l23_e=8, n_h_e=5, tau_ms=20.0, dt_ms=5.0, seed=0,
    )


def _he_factory() -> HE:
    return HE(n_units=5, n_l23_e=8, n_h_pv=2, tau_ms=50.0, dt_ms=5.0, seed=0)


def _hpv_factory() -> HPV:
    return HPV(n_units=2, n_h_e=5, tau_ms=10.0, dt_ms=5.0, seed=0)


POP_FACTORIES: list[tuple[str, Callable, set[str]]] = [
    ("L23E", _l23e_factory, {
        "W_l4_l23_raw", "W_rec_raw", "W_pv_l23_raw",
        "W_som_l23_raw", "W_fb_apical_raw",
    }),
    ("L23PV", _l23pv_factory, {"W_l23_pv_raw"}),
    ("L23SOM", _l23som_factory, {"W_l23_som_raw", "W_fb_som_raw"}),
    ("HE", _he_factory, {"W_l23_h_raw", "W_rec_raw", "W_pv_h_raw"}),
    ("HPV", _hpv_factory, {"W_h_pv_raw"}),
]


@pytest.mark.parametrize("name,factory,expected", POP_FACTORIES)
def test_phase2_all_plastic(name: str, factory: Callable, expected: set[str]) -> None:
    pop = factory()
    pop.set_phase("phase2")
    assert set(pop.plastic_weight_names()) == expected, name
    assert set(pop.frozen_weight_names()) == set(), name


@pytest.mark.parametrize("name,factory,expected", POP_FACTORIES)
def test_phase3_kok_all_frozen(name: str, factory: Callable, expected: set[str]) -> None:
    pop = factory()
    pop.set_phase("phase3_kok")
    assert set(pop.plastic_weight_names()) == set(), name
    assert set(pop.frozen_weight_names()) == expected, name


@pytest.mark.parametrize("name,factory,expected", POP_FACTORIES)
def test_phase3_richter_all_frozen(name: str, factory: Callable, expected: set[str]) -> None:
    pop = factory()
    pop.set_phase("phase3_richter")
    assert set(pop.plastic_weight_names()) == set(), name
    assert set(pop.frozen_weight_names()) == expected, name


def test_plastic_and_frozen_disjoint_all_phases() -> None:
    for name, factory, _ in POP_FACTORIES:
        for phase in ("phase2", "phase3_kok", "phase3_richter"):
            pop = factory()
            pop.set_phase(phase)
            plastic = set(pop.plastic_weight_names())
            frozen = set(pop.frozen_weight_names())
            assert plastic.isdisjoint(frozen), (
                f"{name} @ {phase}: overlap {plastic & frozen}"
            )


def test_plastic_and_frozen_cover_all_weights() -> None:
    for name, factory, expected in POP_FACTORIES:
        for phase in ("phase2", "phase3_kok", "phase3_richter"):
            pop = factory()
            pop.set_phase(phase)
            covered = set(pop.plastic_weight_names()) | set(pop.frozen_weight_names())
            assert covered == expected, (
                f"{name} @ {phase}: missing {expected - covered} "
                f"or extra {covered - expected}"
            )


def test_unknown_phase_raises() -> None:
    pop = _l23pv_factory()
    with pytest.raises(ValueError, match="phase"):
        pop.set_phase("phase4")                                        # type: ignore[arg-type]
