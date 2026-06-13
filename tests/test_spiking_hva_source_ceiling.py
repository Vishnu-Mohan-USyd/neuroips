"""Tests for diagnostic-only spiking-HVA source-ceiling analysis."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "diagnose_spiking_hva_source_ceiling.py"
SPEC = importlib.util.spec_from_file_location("diagnose_spiking_hva_source_ceiling", TOOL_PATH)
assert SPEC is not None
tool = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class SpikingHVASourceCeilingTests(unittest.TestCase):
    def test_train_only_source_ceiling_exports_diagnostic_heldout_and_clip_rows(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            prefix = "synthetic"
            write_csv(
                tmp / f"{prefix}_summary.csv",
                ["metric", "value"],
                [{"metric": "video_clip_length_frames", "value": 4}],
            )

            source_vectors = {
                0: [1.0, 0.0, 0.0, 0.0],
                1: [0.0, 1.0, 0.0, 0.0],
                2: [0.0, 0.0, 1.0, 0.0],
                3: [0.0, 0.0, 0.0, 1.0],
                4: [1.0, 0.0, 0.0, 0.0],
                5: [0.0, 1.0, 0.0, 0.0],
                6: [0.0, 0.0, 1.0, 0.0],
                7: [0.0, 0.0, 0.0, 1.0],
            }
            rate_rows: list[dict[str, object]] = []
            for frame_index, vector in source_vectors.items():
                for tile_id, value in enumerate(vector):
                    rate_rows.append(
                        {
                            "sample_index": frame_index * 4 + tile_id,
                            "repeat_index": 0,
                            "frame_index": frame_index,
                            "tile_id": tile_id,
                            "tile_x": tile_id,
                            "tile_y": 0,
                            "l23e_rate_hz": value * 10.0,
                            "l23e_state_norm": value,
                            "hva_pred_e_spike_rate_hz": value * 3.0,
                            "hva_pred_e_membrane_state_norm": value,
                        }
                    )
            write_csv(
                tmp / f"{prefix}_spiking_hva_rates.csv",
                [
                    "sample_index",
                    "repeat_index",
                    "frame_index",
                    "tile_id",
                    "tile_x",
                    "tile_y",
                    "l23e_rate_hz",
                    "l23e_state_norm",
                    "hva_pred_e_spike_rate_hz",
                    "hva_pred_e_membrane_state_norm",
                ],
                rate_rows,
            )

            prediction_rows: list[dict[str, object]] = []
            prediction_index = 0
            for frame_index, split in [
                (0, "train"),
                (1, "train"),
                (2, "train"),
                (3, "heldout"),  # crosses clip boundary for delay 1 and must be skipped
                (4, "heldout"),
                (5, "heldout"),
                (6, "heldout"),
            ]:
                for tile_id, target in enumerate(source_vectors[frame_index]):
                    prediction_rows.append(
                        {
                            "prediction_index": prediction_index,
                            "repeat_index": 0,
                            "frame_index": frame_index,
                            "target_frame_index": frame_index + 1,
                            "tile_id": tile_id,
                            "split": split,
                            "target_state_norm": target,
                        }
                    )
                    prediction_index += 1
            write_csv(
                tmp / f"{prefix}_spiking_hva_predictions_delay1.csv",
                [
                    "prediction_index",
                    "repeat_index",
                    "frame_index",
                    "target_frame_index",
                    "tile_id",
                    "split",
                    "target_state_norm",
                ],
                prediction_rows,
            )

            output = tool.run_diagnostics(
                genn_dir=tmp,
                prefix=prefix,
                delays=[1],
                ridge_alpha=1.0e-9,
                history_lags=[0, 1],
                clip_length_frames=None,
                local_radius_tiles=2,
                include_predictor_output_sources=False,
                output_path=None,
            )

            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            included_output = tool.run_diagnostics(
                genn_dir=tmp,
                prefix=prefix,
                delays=[1],
                ridge_alpha=1.0e-9,
                history_lags=[0, 1],
                clip_length_frames=None,
                local_radius_tiles=2,
                include_predictor_output_sources=True,
                output_path=tmp / "included_predictor_sources.csv",
            )
            with included_output.open("r", encoding="utf-8", newline="") as handle:
                included_rows = list(csv.DictReader(handle))

        self.assertFalse(any(row["source"].startswith("hva_pred") for row in rows))
        included_pred_rows = [row for row in included_rows if row["source"] == "hva_pred_membrane"]
        self.assertTrue(included_pred_rows)
        self.assertTrue(all(row["predictor_output_state_source"] == "1" for row in included_pred_rows))
        self.assertTrue(all(row["source_prohibited_for_source_guidance"] == "1" for row in included_pred_rows))
        self.assertTrue(all(row["primary_model_success_claim"] == "0" for row in included_pred_rows))

        heldout_global = [
            row
            for row in rows
            if row["delay_frames"] == "1"
            and row["source"] == "l23e_current"
            and row["readout_mode"] == "global_ridge"
            and row["split"] == "heldout"
            and row["clip_id"] == "all"
        ]
        self.assertEqual(len(heldout_global), 1)
        self.assertGreater(float(heldout_global[0]["vector_corr_mean"]), 0.99)
        self.assertEqual(heldout_global[0]["diagnostic_only"], "1")
        self.assertEqual(heldout_global[0]["offline_ridge_ceiling"], "1")
        self.assertEqual(heldout_global[0]["global_offline_ridge"], "1")
        self.assertEqual(heldout_global[0]["architecture_realizable_source_ceiling"], "0")
        self.assertEqual(heldout_global[0]["primary_model_success_claim"], "0")
        self.assertEqual(heldout_global[0]["heldout_updates_applied"], "0")
        self.assertEqual(heldout_global[0]["skipped_cross_clip_count"], "1")

        heldout_local = [
            row
            for row in rows
            if row["delay_frames"] == "1"
            and row["source"] == "l23e_current"
            and row["readout_mode"] == "local_window_ridge"
            and row["split"] == "heldout"
            and row["clip_id"] == "all"
        ]
        self.assertEqual(len(heldout_local), 1)
        self.assertGreater(float(heldout_local[0]["vector_corr_mean"]), 0.99)
        self.assertEqual(heldout_local[0]["global_offline_ridge"], "0")
        self.assertEqual(heldout_local[0]["local_window_readout"], "1")
        self.assertEqual(heldout_local[0]["architecture_realizable_source_ceiling"], "1")
        self.assertEqual(heldout_local[0]["local_radius_tiles"], "2")

        heldout_clip_rows = [
            row
            for row in rows
            if row["source"] == "l23e_current"
            and row["readout_mode"] == "local_window_ridge"
            and row["split"] == "heldout"
            and row["clip_id"] != "all"
        ]
        self.assertEqual(len(heldout_clip_rows), 1)
        self.assertEqual(heldout_clip_rows[0]["clip_id"], "1")
        self.assertEqual(heldout_clip_rows[0]["sample_count"], "3")


if __name__ == "__main__":
    unittest.main()
