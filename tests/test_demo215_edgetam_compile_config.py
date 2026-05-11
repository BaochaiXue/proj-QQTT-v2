from __future__ import annotations

import unittest

from demo_v2_2 import runtime as demo
from demo_v2_1_5 import realtime_three_view_async_filtered_fused_pcd as demo215


class Demo215EdgeTamCompileConfigTest(unittest.TestCase):
    def _contract(self, argv: list[str]) -> dict:
        runtime_argv = demo215._to_demo215_argv(argv)
        parser = demo.build_arg_parser()
        args = parser.parse_args(runtime_argv)
        args = demo.apply_preset_defaults(args, explicit_options=demo.explicit_cli_options(runtime_argv))
        return demo.build_contract(args)

    def test_mask_only_parallel_keeps_mask_only_preset(self) -> None:
        argv = demo215._to_demo215_argv(["--dry-run", "--mask-only-debug", "--parallel-edgetam"])

        self.assertEqual(argv[:2], ["--preset", demo.PRESET_DEMO215_MASK_ONLY_DEBUG])
        self.assertIn("--edgetam-stream-mode", argv)
        self.assertEqual(argv[argv.index("--edgetam-stream-mode") + 1], demo.EDGETAM_STREAM_MODE_PER_CAMERA)
        self.assertIn("--gpu-pipeline-mode", argv)
        self.assertEqual(argv[argv.index("--gpu-pipeline-mode") + 1], demo.GPU_PIPELINE_MODE_STAGED)

    def test_no_compile_maps_to_compile_mode_none(self) -> None:
        argv = demo215._to_demo215_argv(
            ["--dry-run", "--mask-only-debug", "--parallel-edgetam", "--no-compile-edgetam"]
        )

        self.assertEqual(argv[argv.index("--compile-mode") + 1], demo.COMPILE_MODE_NONE)

    def test_towel_prompt_passes_through_public_wrapper(self) -> None:
        argv = demo215._to_demo215_argv(
            [
                "--dry-run",
                "--mask-only-debug",
                "--parallel-edgetam",
                "--controller-object",
                "--object-prompt",
                "stuffed animal",
                "--controller-prompt",
                "towel",
            ]
        )

        self.assertEqual(argv[argv.index("--object-prompt") + 1], "stuffed animal")
        self.assertEqual(argv[argv.index("--controller-prompt") + 1], "towel")

    def test_reduce_overhead_graph_policy_defaults_to_clone(self) -> None:
        contract = self._contract(
            [
                "--dry-run",
                "--mask-only-debug",
                "--parallel-edgetam",
                "--compile-mode",
                demo.COMPILE_MODE_VISION_REDUCE_OVERHEAD,
            ]
        )

        self.assertEqual(contract["edgetam"]["graph_output_policy_effective"], demo.EDGETAM_GRAPH_OUTPUT_POLICY_CLONE)

    def test_no_cudagraph_compile_policy_defaults_to_none(self) -> None:
        contract = self._contract(
            [
                "--dry-run",
                "--mask-only-debug",
                "--parallel-edgetam",
                "--compile-mode",
                demo.COMPILE_MODE_VISION_MAX_AUTOTUNE_NO_CUDAGRAPHS,
            ]
        )

        self.assertEqual(contract["edgetam"]["graph_output_policy_effective"], demo.EDGETAM_GRAPH_OUTPUT_POLICY_NONE)

    def test_compile_scope_components_no_cudagraphs_sets_compile_mode(self) -> None:
        argv = demo215._to_demo215_argv(
            [
                "--dry-run",
                "--mask-only-debug",
                "--parallel-edgetam",
                "--edgetam-compile-scope",
                "components-no-cudagraphs",
            ]
        )
        parser = demo.build_arg_parser()
        args = parser.parse_args(argv)
        args = demo.apply_preset_defaults(args, explicit_options=demo.explicit_cli_options(argv))

        self.assertEqual(args.compile_mode, demo.COMPILE_MODE_COMPONENTS_MAX_AUTOTUNE_NO_CUDAGRAPHS)

    def test_experimental_batch_vision_flag_sets_contract(self) -> None:
        contract = self._contract(["--dry-run", "--experimental-edgetam-batch-vision"])

        self.assertTrue(contract["edgetam"]["batch_vision_encoder"])


if __name__ == "__main__":
    unittest.main()
