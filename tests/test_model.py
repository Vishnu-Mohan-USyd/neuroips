"""Model smoke tests."""

from __future__ import annotations

import math
from pathlib import Path
import unittest

import torch

from v1_snn.analysis import projection_orientation_bias
from v1_snn.config import load_config
from v1_snn.connectivity import build_connectivity
from v1_snn.layout import build_layout
from v1_snn.model import V1TwoLayerSNN
from v1_snn.stimuli import compute_l4_feedforward_drive, generate_grating, make_l4_gabor_bank


ROOT = Path(__file__).resolve().parents[1]


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "configs" / "smoke.toml")
        cls.device = torch.device("cpu")
        cls.layout = build_layout(cls.config, cls.device)

    def test_l4_drive_prefers_matched_orientation(self) -> None:
        gabor_bank = make_l4_gabor_bank(self.config, self.layout, self.device)
        l4 = self.layout.populations["l4_e"]
        neuron_index = int(torch.argmin(torch.remainder(l4.preferred_orientation, torch.pi)).item())
        phase_shifts = torch.linspace(0.0, 2.0 * math.pi, steps=17, device=self.device)[:-1].tolist()
        matched_drives = []
        orthogonal_drives = []
        for phase_rad in phase_shifts:
            matched = generate_grating(
                size_px=self.config.simulation.stimulus_size_px,
                orientation_rad=0.0,
                spatial_frequency_cycles=self.config.simulation.stimulus_spatial_frequency,
                contrast=self.config.simulation.stimulus_contrast,
                phase_rad=phase_rad,
                device=self.device,
            )
            orthogonal = generate_grating(
                size_px=self.config.simulation.stimulus_size_px,
                orientation_rad=0.5 * torch.pi,
                spatial_frequency_cycles=self.config.simulation.stimulus_spatial_frequency,
                contrast=self.config.simulation.stimulus_contrast,
                phase_rad=phase_rad,
                device=self.device,
            )
            matched_drive = compute_l4_feedforward_drive(matched, self.config, self.layout, gabor_bank)
            orthogonal_drive = compute_l4_feedforward_drive(orthogonal, self.config, self.layout, gabor_bank)
            matched_drives.append(float(matched_drive[neuron_index].item()))
            orthogonal_drives.append(float(orthogonal_drive[neuron_index].item()))

        self.assertGreater(max(matched_drives), max(orthogonal_drives))

    def test_feature_biased_projection_has_low_orientation_mismatch(self) -> None:
        projections = build_connectivity(self.config, self.layout, self.device)
        bias = projection_orientation_bias(projections["l4_e_to_l23_e"], self.layout)
        self.assertLess(bias, 0.55)

    def test_protocol_runs_without_nan(self) -> None:
        model = V1TwoLayerSNN(self.config)
        result = model.run_protocol()
        self.assertEqual(result.rates["l4_e"].shape[0], len(self.config.simulation.stimulus_orientations_deg))
        self.assertFalse(torch.isnan(result.rates["l23_e"]).any().item())


if __name__ == "__main__":
    unittest.main()
