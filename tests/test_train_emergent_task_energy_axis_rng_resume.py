"""Regression coverage for CUDA-mapped generator-state resume."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import train_emergent_task_energy_axis as trainer  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_mapped_rng_states_restore_bit_identically(tmp_path: Path) -> None:
    """Restore both arm streams loaded onto CUDA without changing their draws."""

    for resume_function in (trainer.run_pretrain, trainer.run_alpha):
        source = inspect.getsource(resume_function)
        assert source.count("restore_generator_state(") == 2
        assert ".set_state(saved[" not in source

    device = torch.device("cuda:0")
    source_data = trainer.make_generator(device, 400123)
    source_noise = trainer.make_generator(device, 500123)
    torch.rand(137, device=device, generator=source_data)
    torch.rand(211, device=device, generator=source_noise)

    checkpoint_path = tmp_path / "rng_states.pt"
    torch.save(
        {
            "data_generator_state": source_data.get_state(),
            "noise_generator_state": source_noise.get_state(),
        },
        checkpoint_path,
    )
    loaded = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    for state in loaded.values():
        assert isinstance(state, torch.Tensor)
        assert state.device.type == "cuda"
        assert state.dtype == torch.uint8
        assert state.ndim == 1

    expected_data = torch.rand(4096, device=device, generator=source_data)
    expected_noise = torch.rand(4096, device=device, generator=source_noise)

    restored_data = trainer.make_generator(device, 1)
    restored_noise = trainer.make_generator(device, 2)
    trainer.restore_generator_state(
        restored_data,
        loaded["data_generator_state"],
    )
    trainer.restore_generator_state(
        restored_noise,
        loaded["noise_generator_state"],
    )
    actual_data = torch.rand(4096, device=device, generator=restored_data)
    actual_noise = torch.rand(4096, device=device, generator=restored_noise)

    assert torch.equal(actual_data, expected_data)
    assert torch.equal(actual_noise, expected_noise)
    assert torch.equal(restored_data.get_state(), source_data.get_state())
    assert torch.equal(restored_noise.get_state(), source_noise.get_state())

    with pytest.raises(TypeError, match="torch.Tensor"):
        trainer.restore_generator_state(restored_data, b"not a tensor")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="torch.uint8"):
        trainer.restore_generator_state(
            restored_data,
            torch.zeros(8, dtype=torch.int64, device=device),
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        trainer.restore_generator_state(
            restored_data,
            torch.zeros((2, 4), dtype=torch.uint8, device=device),
        )
