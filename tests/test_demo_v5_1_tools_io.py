from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
import unittest


class DemoV51ToolsIoTest(unittest.TestCase):
    def test_load_pickle_and_json_read_standard_artifacts(self) -> None:
        from demo_v5_1.tools.io import load_json, load_pickle

        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            pickle_path = base / "payload.pkl"
            json_path = base / "payload.json"
            expected_pickle = {"values": [1, 2, 3], "name": "demo"}
            expected_json = {"ready": True, "count": 3}

            with pickle_path.open("wb") as handle:
                pickle.dump(expected_pickle, handle)
            json_path.write_text(
                json.dumps(expected_json),
                encoding="utf-8",
            )

            self.assertEqual(expected_pickle, load_pickle(pickle_path))
            self.assertEqual(expected_json, load_json(json_path))


if __name__ == "__main__":
    unittest.main()
