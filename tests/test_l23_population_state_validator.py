"""Tests for the standalone L2/3 population-state validator."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "tools" / "validate_l23_population_state.py"
SPEC = importlib.util.spec_from_file_location("validate_l23_population_state", VALIDATOR_PATH)
assert SPEC is not None
validator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def write_synthetic_video_artifact(directory: Path, prefix: str) -> tuple[Path, Path]:
    """Write a small deterministic repeated-video artifact."""
    site_path = directory / f"{prefix}_video_site_rates.csv"
    frame_path = directory / f"{prefix}_video_frame_summary.csv"
    base_rates = [
        [5.0, 1.0, 0.0, 2.0],
        [0.0, 6.0, 1.0, 3.0],
        [2.0, 0.0, 7.0, 1.0],
        [1.0, 3.0, 0.0, 8.0],
        [4.0, 0.0, 2.0, 6.0],
        [0.0, 5.0, 4.0, 1.0],
    ]

    with site_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("repeat_index,frame_index,population,site_id,rate_hz\n")
        for repeat in range(3):
            gain = 1.0 + (0.01 * repeat)
            for frame, rates in enumerate(base_rates):
                for site, rate in enumerate(rates):
                    handle.write(f"{repeat},{frame},l23e,{site},{rate * gain:.6f}\n")

    with frame_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            "repeat_index,frame_index,frame_start_ms,frame_end_ms,"
            "l4e_rate_hz,l23e_rate_hz,l23pv_rate_hz,l23som_rate_hz,"
            "l4e_drive_min,l4e_drive_mean,l4e_drive_max,l4e_drive_std\n"
        )
        for repeat in range(3):
            for frame in range(len(base_rates)):
                start_ms = float(frame * 100)
                end_ms = start_ms + 100.0
                handle.write(
                    f"{repeat},{frame},{start_ms:.6f},{end_ms:.6f},"
                    "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0\n"
                )

    return site_path, frame_path


class L23PopulationStateValidatorTests(unittest.TestCase):
    def test_synthetic_repeats_compute_expected_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            site_path, frame_path = write_synthetic_video_artifact(tmp, "synthetic")
            artifact = validator.ArtifactInput(
                site_rates_path=site_path,
                frame_summary_path=frame_path,
                expected_frame_summary_path=frame_path,
            )

            result = validator.validate_artifact(
                artifact=artifact,
                population="l23e",
                seed=11,
                shuffle_count=5,
                top_k=2,
                quench_window_frames=2,
                thresholds=[],
            )

            metrics = result["metrics"]
            self.assertEqual(metrics["repeat_count"], 3.0)
            self.assertEqual(metrics["frame_count"], 6.0)
            self.assertEqual(metrics["site_count"], 4.0)
            self.assertAlmostEqual(metrics["repeat_vector_corr_mean"], 1.0)
            self.assertAlmostEqual(metrics["repeat_flat_corr_mean"], 1.0)
            self.assertAlmostEqual(metrics["odd_even_rsm_corr"], 1.0)
            self.assertAlmostEqual(metrics["heldout_decoder_top1_accuracy"], 1.0)
            self.assertIn("fano_mean", metrics)
            self.assertEqual(result["details"]["fano_count_source"], "rate_hz_x_frame_duration")
            self.assertEqual(result["missing_metrics"], [])

    def test_core_crop_filters_row_major_sites(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            site_path = tmp / "crop_video_site_rates.csv"
            site_path.write_text(
                "repeat_index,frame_index,population,site_id,rate_hz\n"
                + "".join(
                    f"{repeat},{frame},l23e,{site},{1.0 + site + frame + repeat:.6f}\n"
                    for repeat in range(2)
                    for frame in range(3)
                    for site in range(16)
                ),
                encoding="utf-8",
            )

            activity = validator.load_population_activity(site_path, "l23e")
            cropped = validator.crop_population_activity(activity, sheet_side=4, core_side=2)

            self.assertEqual(cropped.sites, [5, 6, 9, 10])
            self.assertEqual(cropped.rates_hz.shape, (2, 3, 4))

    def test_missing_repeat_index_is_schema_error(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = Path(raw_tmp) / "bad_video_site_rates.csv"
            path.write_text(
                "frame_index,population,site_id,rate_hz\n"
                "0,l23e,0,1.0\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(validator.InputError, "repeat_index"):
                validator.load_population_activity(path, "l23e")

    def test_fano_reports_missing_frame_summary_without_blocking_other_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            site_path, frame_path = write_synthetic_video_artifact(tmp, "synthetic")
            frame_path.unlink()
            artifact = validator.ArtifactInput(
                site_rates_path=site_path,
                frame_summary_path=None,
                expected_frame_summary_path=frame_path,
            )

            result = validator.validate_artifact(
                artifact=artifact,
                population="l23e",
                seed=3,
                shuffle_count=2,
                top_k=2,
                quench_window_frames=None,
                thresholds=[],
            )

            self.assertAlmostEqual(result["metrics"]["repeat_vector_corr_mean"], 1.0)
            missing = result["missing_metrics"]
            self.assertEqual(len(missing), 1)
            self.assertEqual(missing[0]["metric"], "fano_variability_quenching")
            self.assertEqual(missing[0]["required_file"], str(frame_path))
            self.assertIn("frame_start_ms", missing[0]["required_columns"])


if __name__ == "__main__":
    unittest.main()
