"""Validate the native CUDA module boundary against an exported manifest."""
from __future__ import annotations

import tempfile
from pathlib import Path

from expectation_snn.brian2_model.h_ring import (
    N_CHANNELS as H_N_CHANNELS,
    N_E_PER_CHANNEL as H_N_E_PER_CHANNEL,
    N_INH_POOL,
)
from expectation_snn.brian2_model.v1_ring import (
    N_CHANNELS as V1_N_CHANNELS,
    N_E_PER_CHANNEL as V1_N_E_PER_CHANNEL,
    N_PV_POOL,
    N_SOM_PER_CHANNEL,
)
from expectation_snn.cuda_sim.export_bundle import (
    MANIFEST_SCHEMA_VERSION,
    export_ctx_pred_manifest,
)
from expectation_snn.cuda_sim.native import backend_info, inspect_manifest
from expectation_snn.validation.validate_native_manifest_export import (
    SEED,
    _write_synthetic_checkpoints,
)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="native_module_boundary_") as tmp_s:
        root = Path(tmp_s)
        ckpt_dir = root / "checkpoints"
        manifest_path = root / "ctx_pred_manifest.npz"
        _write_synthetic_checkpoints(ckpt_dir)
        export_ctx_pred_manifest(
            ckpt_dir=ckpt_dir,
            out_path=manifest_path,
            seed=SEED,
            r=1.0,
            g_total=1.0,
            v1_to_h_mode="context_only",
            with_feedback_routes=True,
        )

        info = inspect_manifest(manifest_path)
        pop = info["population_sizes"]
        edges = info["edge_counts"]
        assert info["schema_version"] == MANIFEST_SCHEMA_VERSION
        assert info["synapse_bank_count"] == 18
        assert pop["pop_v1_e_n"] == V1_N_CHANNELS * V1_N_E_PER_CHANNEL
        assert pop["pop_v1_som_n"] == V1_N_CHANNELS * N_SOM_PER_CHANNEL
        assert pop["pop_v1_pv_n"] == N_PV_POOL
        assert pop["pop_ctx_e_n"] == H_N_CHANNELS * H_N_E_PER_CHANNEL
        assert pop["pop_pred_e_n"] == H_N_CHANNELS * H_N_E_PER_CHANNEL
        assert pop["pop_ctx_inh_n"] == N_INH_POOL
        assert pop["pop_pred_inh_n"] == N_INH_POOL
        assert edges["v1_pv_to_e"] == 6144
        assert edges["ctx_ee"] == 36672
        assert edges["pred_ee"] == 36672
        assert edges["ctx_to_pred"] == 36864
        assert edges["v1_to_h_ctx"] == 21504
        assert edges["fb_pred_to_v1e_apical"] == 3072
        assert edges["fb_pred_to_v1som"] == 3072

    print(
        "validate_native_module_boundary: PASS",
        f"backend_info={backend_info()}",
        "synapse_bank_count=18",
        "v1_pv_to_e=6144",
        "ctx_ee=36672",
        "pred_ee=36672",
        "ctx_to_pred=36864",
        "v1_to_h_ctx=21504",
        "fb_direct=3072",
        "fb_som=3072",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

