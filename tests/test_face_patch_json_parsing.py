from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from data_process.visualization.face_quality import parse_face_patches_json


class FacePatchJsonParsingTest(unittest.TestCase):
    def test_parse_name_to_bbox_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "patches.json"
            path.write_text(
                json.dumps(
                    {
                        "0": {
                            "box_front_face": [10, 20, 30, 40],
                            "tabletop_patch": [50, 60, 90, 110],
                        }
                    }
                ),
                encoding="utf-8",
            )
            parsed = parse_face_patches_json(path)
            self.assertIn(0, parsed)
            self.assertEqual(parsed[0][0]["name"], "box_front_face")
            self.assertEqual(tuple(parsed[0][0]["bbox"]), (10, 20, 30, 40))


if __name__ == "__main__":
    unittest.main()
