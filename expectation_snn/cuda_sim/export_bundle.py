"""Export a frozen ctx_pred runtime bundle for the native CUDA simulator.

The native simulator consumes explicit topology, weights, neuron constants,
and checkpoint provenance.  This module deliberately uses the current Brian2
runtime builder as the source of truth so stochastic/implicit topology such as
V1 E->PV is preserved exactly in the exported manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from brian2 import Network, defaultclock, mV, ms, nF, nS, pA, prefs, start_scope
from brian2 import seed as brian_seed

from expectation_snn.assays.runtime import build_frozen_network
from expectation_snn.brian2_model import neurons
from expectation_snn.brian2_model.h_context_prediction import (
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


MANIFEST_SCHEMA_VERSION = 1
DEFAULT_DT_MS = 0.1


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _bytes_json(payload: dict[str, Any]) -> np.bytes_:
    return np.bytes_(json.dumps(_jsonable(payload), sort_keys=True))


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    def run_git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    status = run_git("status", "--short")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "dirty": bool(status),
        "status_short": status or "",
    }


def _q(value: Any, unit: Any) -> np.float64:
    return np.float64(float(value / unit))


def _neuron_constants() -> dict[str, np.ndarray]:
    """Return neuron constants in explicit simulator units."""
    return {
        "const_tau_e_ms": _q(neurons.TAU_E, ms),
        "const_tau_i_ms": _q(neurons.TAU_I, ms),
        "const_tau_nmda_h_ms": _q(neurons.TAU_NMDA_H, ms),
        "const_v_nmda_rev_mV": _q(neurons.V_NMDA_REV, mV),
        "const_v1e_c_soma_nF": _q(neurons.V1E_C_SOMA, nF),
        "const_v1e_gl_soma_nS": _q(neurons.V1E_GL_SOMA, nS),
        "const_v1e_el_mV": _q(neurons.V1E_EL, mV),
        "const_v1e_vt_mV": _q(neurons.V1E_VT, mV),
        "const_v1e_vr_mV": _q(neurons.V1E_VR, mV),
        "const_v1e_refractory_ms": _q(neurons.V1E_REFRACTORY, ms),
        "const_v1e_c_ap_nF": _q(neurons.V1E_C_AP, nF),
        "const_v1e_gl_ap_nS": _q(neurons.V1E_GL_AP, nS),
        "const_v1e_el_ap_mV": _q(neurons.V1E_EL_AP, mV),
        "const_v1e_g_ap_soma_nS": _q(neurons.V1E_G_AP_SOMA, nS),
        "const_v1e_v_ap_th_mV": _q(neurons.V1E_V_AP_TH, mV),
        "const_v1e_v_ap_slope_mV": _q(neurons.V1E_V_AP_SLOPE, mV),
        "const_v1e_a_adapt_nS": _q(neurons.V1E_A_ADAPT, nS),
        "const_v1e_b_adapt_pA": _q(neurons.V1E_B_ADAPT, pA),
        "const_v1e_tau_adapt_ms": _q(neurons.V1E_TAU_ADAPT, ms),
        "const_v1pv_c_nF": _q(neurons.V1PV_C, nF),
        "const_v1pv_gl_nS": _q(neurons.V1PV_GL, nS),
        "const_v1pv_el_mV": _q(neurons.V1PV_EL, mV),
        "const_v1pv_vt_mV": _q(neurons.V1PV_VT, mV),
        "const_v1pv_vr_mV": _q(neurons.V1PV_VR, mV),
        "const_v1pv_refractory_ms": _q(neurons.V1PV_REFRACTORY, ms),
        "const_v1som_c_nF": _q(neurons.V1SOM_C, nF),
        "const_v1som_gl_nS": _q(neurons.V1SOM_GL, nS),
        "const_v1som_el_mV": _q(neurons.V1SOM_EL, mV),
        "const_v1som_vt_mV": _q(neurons.V1SOM_VT, mV),
        "const_v1som_vr_mV": _q(neurons.V1SOM_VR, mV),
        "const_v1som_refractory_ms": _q(neurons.V1SOM_REFRACTORY, ms),
        "const_he_c_nF": _q(neurons.HE_C, nF),
        "const_he_gl_nS": _q(neurons.HE_GL, nS),
        "const_he_el_mV": _q(neurons.HE_EL, mV),
        "const_he_vt_mV": _q(neurons.HE_VT, mV),
        "const_he_vr_mV": _q(neurons.HE_VR, mV),
        "const_he_refractory_ms": _q(neurons.HE_REFRACTORY, ms),
        "const_hinh_c_nF": _q(neurons.HINH_C, nF),
        "const_hinh_gl_nS": _q(neurons.HINH_GL, nS),
        "const_hinh_el_mV": _q(neurons.HINH_EL, mV),
        "const_hinh_vt_mV": _q(neurons.HINH_VT, mV),
        "const_hinh_vr_mV": _q(neurons.HINH_VR, mV),
        "const_hinh_refractory_ms": _q(neurons.HINH_REFRACTORY, ms),
    }


def _scalar_arrays(prefix: str, values: dict[str, int | float | bool]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in values.items():
        if isinstance(value, bool):
            out[f"{prefix}_{key}"] = np.bool_(value)
        elif isinstance(value, int):
            out[f"{prefix}_{key}"] = np.int32(value)
        else:
            out[f"{prefix}_{key}"] = np.float64(value)
    return out


def _synapse_arrays(
    arrays: dict[str, np.ndarray],
    banks: list[dict[str, Any]],
    *,
    name: str,
    syn: Any,
    source: str,
    target: str,
    target_channel: str,
    drive_amp_pA: float,
    nmda_drive_amp_nS: float = 0.0,
) -> None:
    pre = np.asarray(syn.i[:], dtype=np.int32)
    post = np.asarray(syn.j[:], dtype=np.int32)
    w = np.asarray(syn.w[:], dtype=np.float64)
    if pre.shape != post.shape or pre.shape != w.shape:
        raise ValueError(
            f"synapse bank {name!r} shape mismatch: "
            f"pre={pre.shape} post={post.shape} w={w.shape}"
        )
    prefix = f"syn_{name}"
    arrays[f"{prefix}_pre"] = pre
    arrays[f"{prefix}_post"] = post
    arrays[f"{prefix}_w"] = w
    arrays[f"{prefix}_active"] = np.bool_(bool(getattr(syn, "active", True)))
    arrays[f"{prefix}_drive_amp_pA"] = np.float64(drive_amp_pA)
    arrays[f"{prefix}_nmda_drive_amp_nS"] = np.float64(nmda_drive_amp_nS)
    banks.append({
        "name": name,
        "source": source,
        "target": target,
        "target_channel": target_channel,
        "pre_key": f"{prefix}_pre",
        "post_key": f"{prefix}_post",
        "weight_key": f"{prefix}_w",
        "active_key": f"{prefix}_active",
        "drive_amp_pA": float(drive_amp_pA),
        "nmda_drive_amp_nS": float(nmda_drive_amp_nS),
        "n_edges": int(w.size),
        "active": bool(getattr(syn, "active", True)),
    })


def _checkpoint_file_keys(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with np.load(path) as data:
        return sorted(data.files)


def collect_ctx_pred_manifest(
    bundle: Any,
    *,
    repo_root: Path,
    ckpt_dir: Path,
    seed: int,
    r: float,
    g_total: float,
    v1_to_h_mode: str,
    with_feedback_routes: bool,
    dt_ms: float,
) -> dict[str, np.ndarray]:
    """Collect manifest arrays from an already-built frozen ctx_pred bundle."""
    if bundle.ctx_pred is None:
        raise RuntimeError("expected architecture='ctx_pred' bundle")

    v1 = bundle.v1_ring
    ctx = bundle.ctx_pred.ctx
    pred = bundle.ctx_pred.pred
    stage0 = ckpt_dir / f"stage_0_seed{seed}.npz"
    stage1 = ckpt_dir / f"stage_1_ctx_pred_seed{seed}.npz"

    arrays: dict[str, np.ndarray] = {
        "schema_version": np.int32(MANIFEST_SCHEMA_VERSION),
        "dt_ms": np.float64(dt_ms),
        "n_channels": np.int32(V1_N_CHANNELS),
        "h_n_channels": np.int32(H_N_CHANNELS),
        "v1_e_channel": np.asarray(v1.e_channel, dtype=np.int16),
        "v1_som_channel": np.asarray(v1.som_channel, dtype=np.int16),
        "v1_stim_channel": np.asarray(v1.stim_channel, dtype=np.int16),
        "ctx_e_channel": np.asarray(ctx.e_channel, dtype=np.int16),
        "ctx_inh_channel": np.asarray(ctx.inh_channel, dtype=np.int16),
        "pred_e_channel": np.asarray(pred.e_channel, dtype=np.int16),
        "pred_inh_channel": np.asarray(pred.inh_channel, dtype=np.int16),
        "v1_thetas_rad": np.asarray(v1.thetas_rad, dtype=np.float64),
        "ctx_thetas_rad": np.asarray(ctx.thetas_rad, dtype=np.float64),
        "pred_thetas_rad": np.asarray(pred.thetas_rad, dtype=np.float64),
    }
    arrays.update(_neuron_constants())
    arrays.update(_scalar_arrays("pop", {
        "v1_stim_n": int(v1.stim.N),
        "v1_e_n": int(v1.e.N),
        "v1_som_n": int(v1.som.N),
        "v1_pv_n": int(v1.pv.N),
        "ctx_cue_n": int(ctx.cue.N),
        "ctx_e_n": int(ctx.e.N),
        "ctx_inh_n": int(ctx.inh.N),
        "pred_cue_n": int(pred.cue.N),
        "pred_e_n": int(pred.e.N),
        "pred_inh_n": int(pred.inh.N),
        "direction_n": int(bundle.ctx_pred.direction.N),
        "v1_e_per_channel": V1_N_E_PER_CHANNEL,
        "v1_som_per_channel": N_SOM_PER_CHANNEL,
        "v1_pv_pool": N_PV_POOL,
        "h_e_per_channel": H_N_E_PER_CHANNEL,
        "h_inh_pool": N_INH_POOL,
    }))

    banks: list[dict[str, Any]] = []
    v1_cfg = v1.config
    ctx_cfg = ctx.config
    pred_cfg = pred.config
    _synapse_arrays(
        arrays, banks, name="v1_stim_to_e", syn=v1.stim_to_e,
        source="v1_stim", target="v1_e", target_channel="I_e",
        drive_amp_pA=float(v1_cfg.drive_amp_stim_pA),
    )
    _synapse_arrays(
        arrays, banks, name="v1_ee_ring", syn=v1.ee_ring,
        source="v1_e", target="v1_e", target_channel="I_e",
        drive_amp_pA=float(v1_cfg.drive_amp_ee_pA),
    )
    _synapse_arrays(
        arrays, banks, name="v1_e_to_pv", syn=v1.e_to_pv,
        source="v1_e", target="v1_pv", target_channel="I_e",
        drive_amp_pA=float(v1_cfg.drive_amp_e_pv_pA),
    )
    _synapse_arrays(
        arrays, banks, name="v1_pv_to_e", syn=v1.pv_to_e,
        source="v1_pv", target="v1_e", target_channel="I_i",
        drive_amp_pA=float(v1_cfg.drive_amp_pv_e_pA),
    )
    _synapse_arrays(
        arrays, banks, name="v1_e_to_som", syn=v1.e_to_som,
        source="v1_e", target="v1_som", target_channel="I_e",
        drive_amp_pA=float(v1_cfg.drive_amp_e_som_pA),
    )
    _synapse_arrays(
        arrays, banks, name="v1_som_to_e", syn=v1.som_to_e,
        source="v1_som", target="v1_e", target_channel="I_i",
        drive_amp_pA=float(v1_cfg.drive_amp_som_e_pA),
    )
    for prefix, ring, cfg in (("ctx", ctx, ctx_cfg), ("pred", pred, pred_cfg)):
        _synapse_arrays(
            arrays, banks, name=f"{prefix}_ee", syn=ring.ee,
            source=f"{prefix}_e", target=f"{prefix}_e",
            target_channel="I_e,g_nmda_h",
            drive_amp_pA=float(cfg.drive_amp_ee_pA),
            nmda_drive_amp_nS=float(cfg.nmda_drive_amp_nS),
        )
        _synapse_arrays(
            arrays, banks, name=f"{prefix}_e_to_inh", syn=ring.e_to_inh,
            source=f"{prefix}_e", target=f"{prefix}_inh",
            target_channel="I_e",
            drive_amp_pA=float(cfg.drive_amp_e_inh_pA),
        )
        _synapse_arrays(
            arrays, banks, name=f"{prefix}_inh_to_e", syn=ring.inh_to_e,
            source=f"{prefix}_inh", target=f"{prefix}_e",
            target_channel="I_i",
            drive_amp_pA=float(cfg.drive_amp_inh_e_pA),
        )
        _synapse_arrays(
            arrays, banks, name=f"{prefix}_cue_to_e", syn=ring.cue_to_e,
            source=f"{prefix}_cue", target=f"{prefix}_e",
            target_channel="I_e",
            drive_amp_pA=float(cfg.drive_amp_cue_e_pA),
        )

    if bundle.v1_to_h is not None:
        _synapse_arrays(
            arrays, banks, name="v1_to_h_ctx", syn=bundle.v1_to_h.v1_to_he,
            source="v1_e", target="ctx_e", target_channel="I_e",
            drive_amp_pA=float(bundle.v1_to_h.config.drive_amp_v1_to_h_pA),
        )
        arrays["v1_to_h_kernel"] = np.asarray(bundle.v1_to_h.kernel, dtype=np.float64)
        arrays["v1_to_h_kernel_w"] = np.asarray(
            bundle.v1_to_h.kernel_w, dtype=np.float64,
        )

    _synapse_arrays(
        arrays, banks, name="ctx_to_pred", syn=bundle.ctx_pred.ctx_pred,
        source="ctx_e", target="pred_e", target_channel="I_e",
        drive_amp_pA=float(bundle.ctx_pred.config.drive_amp_ctx_pred_pA),
    )
    _synapse_arrays(
        arrays, banks, name="fb_pred_to_v1e_apical", syn=bundle.fb.hr_to_v1e,
        source="pred_e", target="v1_e", target_channel="I_ap_e",
        drive_amp_pA=float(bundle.fb.config.drive_amp_h_to_v1e_apical_pA),
    )
    _synapse_arrays(
        arrays, banks, name="fb_pred_to_v1som", syn=bundle.fb.hr_to_v1som,
        source="pred_e", target="v1_som", target_channel="I_e",
        drive_amp_pA=float(bundle.fb.config.drive_amp_h_to_v1som_pA),
    )
    arrays["fb_kernel_direct"] = np.asarray(bundle.fb.kernel_direct, dtype=np.float64)
    arrays["fb_kernel_som"] = np.asarray(bundle.fb.kernel_som, dtype=np.float64)
    arrays["fb_kernel_w_direct"] = np.asarray(bundle.fb.kernel_w_direct, dtype=np.float64)
    arrays["fb_kernel_w_som"] = np.asarray(bundle.fb.kernel_w_som, dtype=np.float64)

    arrays["ckpt_stage0_pv_to_e_w_loaded"] = arrays["syn_v1_pv_to_e_w"].copy()
    arrays["ckpt_stage1_ctx_ee_w_loaded"] = arrays["syn_ctx_ee_w"].copy()
    arrays["ckpt_stage1_pred_ee_w_loaded"] = arrays["syn_pred_ee_w"].copy()
    arrays["ckpt_stage1_ctx_pred_w_loaded"] = arrays["syn_ctx_to_pred_w"].copy()

    arrays["ctx_pred_config_json"] = np.bytes_(
        h_context_prediction_config_to_json(bundle.ctx_pred.config),
    )

    metadata = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dt_ms": float(dt_ms),
        "seed": int(seed),
        "architecture": "ctx_pred",
        "repo_root": str(repo_root),
        "git": _git_provenance(repo_root),
        "checkpoint": {
            "ckpt_dir": str(ckpt_dir),
            "stage0_path": str(stage0),
            "stage1_ctx_pred_path": str(stage1),
            "stage0_sha256": _sha256_file(stage0),
            "stage1_ctx_pred_sha256": _sha256_file(stage1),
            "stage0_keys": _checkpoint_file_keys(stage0),
            "stage1_ctx_pred_keys": _checkpoint_file_keys(stage1),
        },
        "population_sizes": {
            key.removeprefix("pop_"): int(value)
            for key, value in arrays.items()
            if key.startswith("pop_")
        },
        "runtime": {
            "r": float(r),
            "g_total": float(g_total),
            "g_direct": float(bundle.fb.g_direct),
            "g_SOM": float(bundle.fb.g_SOM),
            "with_v1_to_h": str(v1_to_h_mode),
            "with_feedback_routes": bool(with_feedback_routes),
            "v1_pv_bg_enabled": bool(v1_cfg.pv_bg_enabled),
            "v1_pv_bg_rate_hz": float(v1_cfg.pv_bg_rate_hz),
            "v1_pv_bg_weight_pA": float(v1_cfg.pv_bg_weight_pA),
        },
        "bundle_meta": _jsonable(bundle.meta),
        "configs": {
            "v1": _jsonable(v1_cfg),
            "ctx": _jsonable(ctx_cfg),
            "pred": _jsonable(pred_cfg),
            "ctx_pred": json.loads(
                h_context_prediction_config_to_json(bundle.ctx_pred.config),
            ),
            "feedback": _jsonable(bundle.fb.config),
            "v1_to_h": (
                None if bundle.v1_to_h is None
                else _jsonable(bundle.v1_to_h.config)
            ),
        },
        "synapse_banks": banks,
    }
    arrays["metadata_json"] = _bytes_json(metadata)
    arrays["synapse_banks_json"] = _bytes_json({"synapse_banks": banks})
    return arrays


def export_ctx_pred_manifest(
    *,
    ckpt_dir: str | Path,
    out_path: str | Path,
    seed: int = 42,
    r: float = 1.0,
    g_total: float = 1.0,
    v1_to_h_mode: str = "context_only",
    with_feedback_routes: bool = True,
    dt_ms: float = DEFAULT_DT_MS,
    reset_scope: bool = True,
) -> Path:
    """Build and export the frozen ctx_pred manifest to an NPZ file."""
    repo_root = Path(__file__).resolve().parents[2]
    ckpt_dir = Path(ckpt_dir)
    out_path = Path(out_path)
    if reset_scope:
        start_scope()
    brian_seed(int(seed))
    np.random.seed(int(seed))
    prefs.codegen.target = "numpy"
    defaultclock.dt = float(dt_ms) * ms

    bundle = build_frozen_network(
        architecture="ctx_pred",
        seed=int(seed),
        r=float(r),
        g_total=float(g_total),
        with_cue=False,
        with_v1_to_h=str(v1_to_h_mode),
        with_feedback_routes=bool(with_feedback_routes),
        ckpt_dir=str(ckpt_dir),
    )
    # Register all Brian2 objects and run zero simulated time.  This keeps the
    # export path side-effect free while making Brian2 mark the objects as used.
    _net = Network(*bundle.groups)
    _net.run(0 * ms)
    manifest = collect_ctx_pred_manifest(
        bundle,
        repo_root=repo_root,
        ckpt_dir=ckpt_dir,
        seed=int(seed),
        r=float(r),
        g_total=float(g_total),
        v1_to_h_mode=str(v1_to_h_mode),
        with_feedback_routes=bool(with_feedback_routes),
        dt_ms=float(dt_ms),
    )
    _ = _net
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **manifest)
    return out_path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--r", type=float, default=1.0)
    parser.add_argument("--g-total", type=float, default=1.0)
    parser.add_argument(
        "--v1-to-h-mode",
        choices=("continuous", "context_only", "off"),
        default="context_only",
    )
    parser.add_argument("--disable-feedback-routes", action="store_true")
    parser.add_argument("--dt-ms", type=float, default=DEFAULT_DT_MS)
    args = parser.parse_args(list(argv) if argv is not None else None)

    path = export_ctx_pred_manifest(
        ckpt_dir=args.ckpt_dir,
        out_path=args.out,
        seed=args.seed,
        r=args.r,
        g_total=args.g_total,
        v1_to_h_mode=args.v1_to_h_mode,
        with_feedback_routes=not args.disable_feedback_routes,
        dt_ms=args.dt_ms,
    )
    with np.load(path) as data:
        banks = json.loads(bytes(data["synapse_banks_json"]).decode("utf-8"))[
            "synapse_banks"
        ]
    print(
        "export_ctx_pred_manifest:",
        f"wrote={path}",
        f"schema={MANIFEST_SCHEMA_VERSION}",
        f"synapse_banks={len(banks)}",
    )
    for bank in banks:
        print(
            "export_ctx_pred_manifest:",
            f"{bank['name']} n_edges={bank['n_edges']} active={bank['active']}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
