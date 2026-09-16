import unittest

from phase0_dataset_management_testing.p0_config import CONFIG
from phase0_dataset_management_testing.p0_logging import setup_logger
from phase0_dataset_management_testing.p0_scan_datasets import build_device_manifest


class ScanDeviceJsonTest(unittest.TestCase):
    """
    Only one device ECG JSON structure is accepted: a list of packet
    records (admissionId + nested value[] array) — see the module
    docstring in p0_scan_datasets.py. Confirmed identical across the device
    fixtures included here (e.g. sessions ADM275758535 and ADM1345459698),
    so a single sample per device family
    in the test fixtures is enough to cover the format.
    """

    def setUp(self):
        cfg = CONFIG.copy()
        cfg["use_test_device_path"] = True
        self.manifest = build_device_manifest(cfg, logger=None)

    def test_packetized_ecg_sessions_are_scanned(self):
        for session_id in ("ADM1345459698", "ADM275758535"):
            self.assertIn(session_id, self.manifest)
            chunk = self.manifest[session_id][0]
            self.assertEqual(chunk["session_id"], session_id)
            self.assertGreater(chunk["sample_count"], 0)
            self.assertIsNotNone(chunk["window_start_ms"])
            self.assertIsNotNone(chunk["window_end_ms"])

    def test_non_ecg_alert_files_are_never_registered_as_sessions(self):
        # ADM1189416147's only JSON file in the fixtures is a device alert
        # record ({"category": "alert", "streamAlert": [...]}), not an ECG
        # chunk — it must never surface as a scannable session.
        self.assertNotIn("ADM1189416147", self.manifest)

    def test_phase0_logger_accepts_log_file_keyword(self):
        logger = setup_logger(
            "phase0_test_logger_regression",
            log_dir="logs",
            log_file="phase0_test_logger_regression.log",
        )
        self.assertIsNotNone(logger)


if __name__ == "__main__":
    unittest.main()