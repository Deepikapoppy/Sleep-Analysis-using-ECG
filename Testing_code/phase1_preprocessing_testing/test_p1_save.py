import logging
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Testing_code.phase1_preprocessing_testing.p1_save import save_preprocessed


class SavePreprocessedDiskSpaceTests(unittest.TestCase):
    def test_saves_outputs_when_free_space_is_low(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "output_dir": tmpdir,
                "record_name": "test_record",
                "dataset": "device",
                "epoch_sec": 30,
            }
            logger = logging.getLogger("test_save_preprocessed")
            logger.handlers.clear()
            logger.addHandler(logging.NullHandler())

            epochs = np.ones((4000, 2000), dtype=np.float64)
            bad_mask = np.zeros(len(epochs), dtype=bool)
            clean_ecg = np.ones(200000, dtype=np.float64)
            raw_ecg_ds = np.ones(200000, dtype=np.float64)
            sqi_df = pd.DataFrame({"overall_sqi": np.linspace(0.5, 0.9, len(epochs))})

            with patch("shutil.disk_usage", return_value=SimpleNamespace(free=60_000_000)):
                meta = save_preprocessed(
                    epochs,
                    bad_mask,
                    clean_ecg,
                    raw_ecg_ds,
                    sqi_df,
                    fs=125.0,
                    config=config,
                    logger=logger,
                )

            self.assertEqual(meta["record"], "test_record")
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "preprocessed_epochs.npy")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "clean_ecg_full.npy")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "phase1_meta.json")))


if __name__ == "__main__":
    unittest.main()
