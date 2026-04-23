"""Validate native CUDA manifest export with synthetic ctx_pred checkpoints."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from expectation_snn.brian2_model.h_context_prediction import (
    H_CONTEXT_PREDICTION_CONFIG_SCHEMA_VERSION,
    HContextPredictionConfig,
    h_context_prediction_config_to_json,
)
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


SEED = 42


def _write_synthetic_checkpoints(ckpt_dir: Path) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    n_v1_e = V1_N_CHANNELS * V1_N_E_PER_CHANNEL
    n_h_e = H_N_CHANNELS * H_N_E_PER_CHANNEL
    n_v1_pv_to_e = N_PV_POOL * n_v1_e
    n_h_ee = n_h_e * (n_h_e - 1)
    n_ctx_pred = n_h_e * n_h_e

    cfg = HContextPredictionConfig(
        drive_amp_ctx_pred_pA=25.0,
        pred_e_uniform_bias_pA=0.0,
    )
    cfg_json = h_context_prediction_config_to_json(cfg)
    np.savez(
        ckpt_dir / f"stage_0_seed{SEED}.npz",
        bias_pA=np.float64(0.0),
        pv_to_e_w=np.linspace(0.0, 0.25, n_v1_pv_to_e, dtype=np.float64),
    )
    np.savez(
        ckpt_dir / f"stage_1_ctx_pred_seed{SEED}.npz",
        ctx_ee_w_final=np.linspace(0.0, 0.30, n_h_ee, dtype=np.float64),
        pred_ee_w_final=np.linspace(0.0, 0.20, n_h_ee, dtype=np.float64),
        W_ctx_pred_final=np.full((n_ctx_pred,), 0.005, dtype=np.float64),
        ctx_pred_config_schema_version=np.int32(
            H_CONTEXT_PREDICTION_CONFIG_SCHEMA_VERSION,
        ),
        ctx_pred_config_json=np.bytes_(cfg_json),
        ctx_pred_drive_amp_ctx_pred_pA=np.float64(cfg.drive_amp_ctx_pred_pA),
        ctx_pred_pred_e_uniform_bias_pA=np.float64(cfg.pred_e_uniform_bias_pA),
    )


def _metadata(data: np.lib.npyio.NpzFile) -> dict:
    return json.loads(bytes(data["metadata_json"]).decode("utf-8"))


def _n_edges(data: np.lib.npyio.NpzFile, name: str) -> int:
    pre = data[f"syn_{name}_pre"]
    post = data[f"syn_{name}_post"]
    w = data[f"syn_{name}_w"]
    assert pre.shape == post.shape == w.shape, (name, pre.shape, post.shape, w.shape)
    return int(pre.shape[0])


def _synapse_array_keys(data: np.lib.npyio.NpzFile) -> list[str]:
    return sorted(
        key for key in data.files
        if key.startswith("syn_")
        and key.endswith(("_pre", "_post", "_w"))
    )


def _assert_repeat_export_reproducible(path_a: Path, path_b: Path) -> None:
    with np.load(path_a) as a, np.load(path_b) as b:
        keys_a = _synapse_array_keys(a)
        keys_b = _synapse_array_keys(b)
        assert keys_a == keys_b
        for suffix in ("pre", "post", "w"):
            assert f"syn_v1_e_to_pv_{suffix}" in keys_a
        for key in keys_a:
            assert np.array_equal(a[key], b[key]), key


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="native_manifest_export_") as tmp_s:
        root = Path(tmp_s)
        ckpt_dir = root / "checkpoints"
        out_path = root / "ctx_pred_manifest_a.npz"
        out_path_b = root / "ctx_pred_manifest_b.npz"
        _write_synthetic_checkpoints(ckpt_dir)

        export_ctx_pred_manifest(
            ckpt_dir=ckpt_dir,
            out_path=out_path,
            seed=SEED,
            r=1.0,
            g_total=1.0,
            v1_to_h_mode="context_only",
            with_feedback_routes=True,
        )
        export_ctx_pred_manifest(
            ckpt_dir=ckpt_dir,
            out_path=out_path_b,
            seed=SEED,
            r=1.0,
            g_total=1.0,
            v1_to_h_mode="context_only",
            with_feedback_routes=True,
        )
        _assert_repeat_export_reproducible(out_path, out_path_b)

        with np.load(out_path) as data:
            meta = _metadata(data)
            assert int(data["schema_version"]) == MANIFEST_SCHEMA_VERSION
            assert float(data["dt_ms"]) == 0.1
            assert int(data["pop_v1_e_n"]) == V1_N_CHANNELS * V1_N_E_PER_CHANNEL
            assert int(data["pop_v1_som_n"]) == V1_N_CHANNELS * N_SOM_PER_CHANNEL
            assert int(data["pop_v1_pv_n"]) == N_PV_POOL
            assert int(data["pop_ctx_e_n"]) == H_N_CHANNELS * H_N_E_PER_CHANNEL
            assert int(data["pop_pred_e_n"]) == H_N_CHANNELS * H_N_E_PER_CHANNEL
            assert int(data["pop_ctx_inh_n"]) == N_INH_POOL
            assert int(data["pop_pred_inh_n"]) == N_INH_POOL

            assert _n_edges(data, "v1_pv_to_e") == 6144
            assert _n_edges(data, "ctx_ee") == 36672
            assert _n_edges(data, "pred_ee") == 36672
            assert _n_edges(data, "ctx_to_pred") == 36864
            assert _n_edges(data, "v1_to_h_ctx") == 21504
            assert _n_edges(data, "fb_pred_to_v1e_apical") == 3072
            assert _n_edges(data, "fb_pred_to_v1som") == 3072

            assert data["ckpt_stage0_pv_to_e_w_loaded"].shape == (6144,)
            assert data["ckpt_stage1_ctx_ee_w_loaded"].shape == (36672,)
            assert data["ckpt_stage1_pred_ee_w_loaded"].shape == (36672,)
            assert data["ckpt_stage1_ctx_pred_w_loaded"].shape == (36864,)
            assert bool(data["syn_v1_pv_to_e_active"]) is False

            assert meta["schema_version"] == MANIFEST_SCHEMA_VERSION
            assert meta["runtime"]["with_v1_to_h"] == "context_only"
            assert meta["runtime"]["g_direct"] == 0.5
            assert meta["runtime"]["g_SOM"] == 0.5
            assert meta["bundle_meta"]["architecture"] == "ctx_pred"
            assert meta["bundle_meta"]["ctx_pred_config_source"] == "checkpoint"
            assert len(meta["checkpoint"]["stage0_sha256"]) == 64
            assert len(meta["checkpoint"]["stage1_ctx_pred_sha256"]) == 64

    print(
        "validate_native_manifest_export: PASS",
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
