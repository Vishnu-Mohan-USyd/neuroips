from __future__ import annotations

import unittest

import torch

from lvc_expectation.geometry import OrientationGeometry


class GeometryTests(unittest.TestCase):
    def test_orientation_geometry_wraps_at_180_degrees(self) -> None:
        geometry = OrientationGeometry(n_orientations=12)
        self.assertEqual(geometry.circular_distance_bins(0, 11), 1)
        self.assertEqual(geometry.circular_distance_deg(0, 6), 90.0)
        self.assertEqual(geometry.step_deg, 15.0)
        self.assertIsInstance(geometry.circular_distance_bins(0, 11), int)

    def test_circular_shift_is_equivariant(self) -> None:
        geometry = OrientationGeometry(n_orientations=12)
        values = torch.arange(12)
        shifted = geometry.circular_shift(values, shift=2)
        self.assertEqual(shifted.tolist(), [10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_gaussian_kernel_is_normalized(self) -> None:
        geometry = OrientationGeometry(n_orientations=12)
        kernel = geometry.gaussian_kernel(center=0, width_deg=30.0)
        self.assertTrue(torch.isclose(kernel.sum(), torch.tensor(1.0)))
        self.assertEqual(tuple(kernel.shape), (12,))

    def test_gaussian_kernel_rotates_with_center(self) -> None:
        geometry = OrientationGeometry(n_orientations=12)
        kernel_a = geometry.gaussian_kernel(center=0, width_deg=30.0)
        kernel_b = geometry.gaussian_kernel(center=3, width_deg=30.0)
        self.assertTrue(torch.allclose(kernel_b, geometry.circular_shift(kernel_a, shift=3)))

    def test_shifted_indices_match_roll_contract(self) -> None:
        geometry = OrientationGeometry(n_orientations=12)
        self.assertEqual(geometry.shifted_indices(2).tolist(), [10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_circular_distance_bins_accepts_tensor_inputs(self) -> None:
        geometry = OrientationGeometry(n_orientations=12)
        src = torch.tensor([[1, 2]])
        dst = torch.tensor([[2, 3]])
        distances = geometry.circular_distance_bins(src, dst)
        self.assertTrue(torch.equal(distances, torch.tensor([[1, 1]])))

    def test_circular_distance_deg_accepts_mixed_scalar_and_tensor_inputs(self) -> None:
        geometry = OrientationGeometry(n_orientations=12)
        distances = geometry.circular_distance_deg(0, torch.tensor([0, 1, 11]))
        self.assertTrue(torch.equal(distances, torch.tensor([0.0, 15.0, 15.0])))


if __name__ == "__main__":
    unittest.main()
