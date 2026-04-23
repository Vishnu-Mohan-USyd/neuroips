"""Brian2 backend selection for expectation_snn.

The default backend remains Brian2's NumPy target. Standalone backends are
opt-in via EXPECTATION_SNN_BACKEND so import-only validators and current CPU
scripts keep their existing behavior unless explicitly configured otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from brian2 import prefs, set_device


BACKEND_ENV = "EXPECTATION_SNN_BACKEND"
STANDALONE_DIR_ENV = "EXPECTATION_SNN_STANDALONE_DIR"
DEFAULT_BACKEND = "numpy"
ALLOWED_BACKENDS = ("numpy", "cpp_standalone", "cuda_standalone", "cuda")


@dataclass(frozen=True)
class BackendConfig:
    """Resolved Brian2 backend configuration."""

    name: str
    directory: Path | None
    build_on_run: bool


def selected_backend() -> str:
    """Return the normalized backend name requested by the environment."""
    raw = os.environ.get(BACKEND_ENV, DEFAULT_BACKEND).strip().lower()
    if raw not in ALLOWED_BACKENDS:
        allowed = ", ".join(ALLOWED_BACKENDS)
        raise ValueError(f"{BACKEND_ENV}={raw!r} is invalid; allowed values: {allowed}")
    if raw == "cuda":
        return "cuda_standalone"
    return raw


def _standalone_directory(backend: str, directory: str | os.PathLike[str] | None) -> Path:
    if directory is not None:
        return Path(directory).expanduser()
    env_dir = os.environ.get(STANDALONE_DIR_ENV)
    if env_dir:
        return Path(env_dir).expanduser()
    return Path("/tmp") / "expectation_snn_brian2" / f"{backend}_{os.getpid()}"


def configure_backend(
    *,
    build_on_run: bool = True,
    directory: str | os.PathLike[str] | None = None,
) -> BackendConfig:
    """Configure Brian2 according to EXPECTATION_SNN_BACKEND.

    Parameters
    ----------
    build_on_run
        Passed to Brian2 standalone devices. The default is True so tiny smoke
        validators can run without an explicit ``device.build`` call.
    directory
        Optional standalone build directory. If omitted, uses
        EXPECTATION_SNN_STANDALONE_DIR when set, otherwise a process-specific
        directory under ``/tmp``.
    """
    backend = selected_backend()
    if backend == "numpy":
        prefs.codegen.target = "numpy"
        print("expectation_snn backend: numpy (prefs.codegen.target='numpy')")
        return BackendConfig(name=backend, directory=None, build_on_run=False)

    build_dir = _standalone_directory(backend, directory)
    build_dir.parent.mkdir(parents=True, exist_ok=True)
    if backend == "cpp_standalone":
        set_device("cpp_standalone", directory=str(build_dir), build_on_run=build_on_run)
    elif backend == "cuda_standalone":
        import brian2cuda  # noqa: F401  # Registers Brian2's cuda_standalone device.

        set_device("cuda_standalone", directory=str(build_dir), build_on_run=build_on_run)
    else:  # selected_backend validates/normalizes before this point.
        raise AssertionError(f"Unhandled backend {backend!r}")
    print(
        f"expectation_snn backend: {backend} "
        f"(directory={build_dir}, build_on_run={build_on_run})"
    )
    return BackendConfig(name=backend, directory=build_dir, build_on_run=build_on_run)
