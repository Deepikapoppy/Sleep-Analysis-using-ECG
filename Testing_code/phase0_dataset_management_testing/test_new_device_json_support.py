import unittest

from phase0_dataset_management_testing.p0_config import CONFIG
from phase0_dataset_management_testing.p0_logging import setup_logger
from phase0_dataset_management_testing.p0_scan_datasets import build_device_manifest


class NewDeviceJsonSupportTest(unittest.TestCase):
    def test_scan_datasets_supports_legacy_and_new_json_formats(self):
        cfg = CONFIG.copy()
        cfg["use_test_device_path"] = True
        manifest = build_device_manifest(cfg, logger=None)

        self.assertIn("ADM1345459698", manifest)
        self.assertIn("ADM275758535", manifest)
        self.assertIn("ADM1189416147", manifest)

        first = manifest["ADM1345459698"][0]
        self.assertEqual(first["session_id"], "ADM1345459698")
        self.assertGreater(first["sample_count"], 0)
        self.assertIsNotNone(first["window_start_ms"])
        self.assertIsNotNone(first["window_end_ms"])

        legacy = manifest["ADM1189416147"][0]
        self.assertEqual(legacy["session_id"], "ADM1189416147")
        self.assertGreater(legacy["sample_count"], 0)

    def test_phase0_logger_accepts_log_file_keyword(self):
        logger = setup_logger(
            "phase0_test_logger_regression",
            log_dir="logs",
            log_file="phase0_test_logger_regression.log",
        )
        self.assertIsNotNone(logger)


if __name__ == "__main__":
    unittest.main()
