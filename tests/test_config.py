"""Configuration loading tests."""

from __future__ import annotations

from pathlib import Path
import unittest

from v1_snn.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigLoadTests(unittest.TestCase):
    def test_load_smoke_config(self) -> None:
        config = load_config(ROOT / "configs" / "smoke.toml")
        self.assertEqual(config.sheet.side, 8)
        self.assertEqual(config.populations.l4_e_per_site, 6)
        self.assertIn("l4_e_to_l23_e", config.connectivity)


if __name__ == "__main__":
    unittest.main()
