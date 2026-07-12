from __future__ import annotations

from contextlib import redirect_stderr
import io
import unittest

from demo_v6_2 import main_cli
from demo_v6_2 import mdp_cli
from demo_v6_2.main_subprocess import _contract


class RuntimeInputModeTests(unittest.TestCase):
    def test_orchestrator_accepts_only_fake_live_and_live(self) -> None:
        parser = main_cli.build_parser()
        supported_modes = {"fake-live", "live"}
        input_source_action = next(
            action for action in parser._actions if action.dest == "input_source"
        )
        self.assertEqual(set(input_source_action.choices or ()), supported_modes)
        self.assertIn(parser.parse_args([]).input_source, supported_modes)

        for input_source in supported_modes:
            args = parser.parse_args(["--input-source", input_source])
            self.assertEqual(args.input_source, input_source)
            self.assertFalse(hasattr(args, "source_headless_capture"))
        self.assertNotIn("source_headless_capture", _contract(parser.parse_args([])))

        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                parser.parse_args(["--source-headless-capture", "/tmp/capture"])
        self.assertEqual(error.exception.code, 2)

    def test_camera_runtime_accepts_only_fake_live_and_live(self) -> None:
        parser = mdp_cli.build_parser()
        supported_modes = {"fake-live", "live"}
        input_source_action = next(
            action for action in parser._actions if action.dest == "input_source"
        )
        self.assertEqual(set(input_source_action.choices or ()), supported_modes)
        self.assertIn(parser.parse_args([]).input_source, supported_modes)

        for input_source in supported_modes:
            args = parser.parse_args(["--input-source", input_source])
            self.assertEqual(args.input_source, input_source)

        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                parser.parse_args(["--input-source", "recording"])
        self.assertEqual(error.exception.code, 2)

        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                parser.parse_args(["--recording-case", "/tmp/case"])
        self.assertEqual(error.exception.code, 2)

        self.assertTrue(mdp_cli._is_replay_input_source("fake-live"))
        self.assertFalse(mdp_cli._is_replay_input_source("live"))
        self.assertFalse(mdp_cli._is_replay_input_source("recording"))


if __name__ == "__main__":
    unittest.main()
