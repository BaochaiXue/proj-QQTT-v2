from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

from qqtt.demo import demo31_runtime as demo31
from qqtt.demo import demo3_runtime


PRESET_DEMO32_FFS_TAPNEXTPP = demo31.PRESET_DEMO32_FFS_TAPNEXTPP
PRESET_DEMO32_FFS_LITETRACKER = demo31.PRESET_DEMO32_FFS_LITETRACKER
DEMO32_RUNTIME_MODULE = "qqtt.demo.demo32_runtime"
DEMO32_RUNTIME_OWNER = "demo32_tracker_ffs"


def build_arg_parser() -> argparse.ArgumentParser:
    return demo31.build_arg_parser(default_preset=PRESET_DEMO32_FFS_TAPNEXTPP)


def apply_preset_defaults(
    args: argparse.Namespace,
    *,
    explicit_options: set[str] | None = None,
) -> argparse.Namespace:
    return demo31.apply_preset_defaults(args, explicit_options=explicit_options)


def validate_args(
    args: argparse.Namespace,
    *,
    require_calibration: bool = False,
    cuda_device_count_provider: demo31.CudaDeviceCountProvider | None = None,
) -> None:
    demo31.validate_args(
        args,
        require_calibration=require_calibration,
        cuda_device_count_provider=cuda_device_count_provider,
    )


def validate_live_contract(
    args: argparse.Namespace,
    *,
    connected_serials_provider: demo31.ConnectedSerialsProvider | None = None,
    cuda_device_count_provider: demo31.CudaDeviceCountProvider | None = None,
) -> dict[str, Any]:
    return demo31.validate_live_realsense_contract(
        args,
        connected_serials_provider=connected_serials_provider,
        cuda_device_count_provider=cuda_device_count_provider,
    )


def build_contract(
    args: argparse.Namespace,
    *,
    cuda_device_count_provider: demo31.CudaDeviceCountProvider | None = None,
) -> dict[str, Any]:
    contract = demo31.build_contract(args, cuda_device_count_provider=cuda_device_count_provider)
    contract.update(
        {
            "demo": "demo3.2",
            "runtime_module": DEMO32_RUNTIME_MODULE,
            "runtime_owner": DEMO32_RUNTIME_OWNER,
            "independent_demo_runtime": True,
            "derived_from_demo31_preset": False,
            "delegates_to_demo23_entrypoint": False,
            "shared_runtime_services_reused": True,
            "tracker_result_required_for_render": True,
            "tracker_marker_required_for_render": True,
            "tracker_input_publish_hooks": ["raw_fused_async", "fused_packet"],
        }
    )
    contract["profile_summary_fields"] = demo31.build_empty_dual_gpu_profile_summary(contract)
    return contract


def format_contract(contract: dict[str, Any]) -> str:
    prefix_keys = (
        "runtime_module",
        "runtime_owner",
        "independent_demo_runtime",
        "derived_from_demo31_preset",
        "delegates_to_demo23_entrypoint",
        "tracker_result_required_for_render",
        "tracker_marker_required_for_render",
        "tracker_input_publish_hooks",
    )
    prefix = []
    for key in prefix_keys:
        value = contract[key]
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        prefix.append(f"{key} = {rendered}")
    return "\n".join([*prefix, demo31.format_contract(contract)])


def build_shared_runtime_args(
    args: argparse.Namespace,
    *,
    shared_runtime_module: Any | None,
    live_validation: dict[str, Any],
    shared_profile_path: Path | None,
) -> argparse.Namespace:
    shared_args = demo31.build_shared_runtime_args(
        args,
        shared_runtime_module=shared_runtime_module,
        live_validation=live_validation,
        shared_profile_path=shared_profile_path,
    )
    shared_args.demo_version_override = "demo3.2"
    shared_args.demo_display_name_override = "Demo 3.2"
    shared_args.demo32_independent_runtime = True
    return shared_args


def make_demo32_live_runtime_class(
    shared_runtime_module: Any,
    *,
    process_client_factory: demo31.ProcessClientFactory | None = None,
):
    base_cls = demo31.make_demo31_live_runtime_class(
        shared_runtime_module,
        process_client_factory=process_client_factory,
    )

    class Demo32LiveRuntime(base_cls):
        """Demo 3.2-owned live runtime for FFS + point-tracker control markers."""

    Demo32LiveRuntime.__name__ = "Demo32LiveRuntime"
    return Demo32LiveRuntime


class Demo32Runtime(demo31.Demo31Runtime):
    def __init__(
        self,
        args: argparse.Namespace,
        *,
        shared_runtime_module: Any | None = None,
        shared_runtime_cls: type | None = None,
        connected_serials_provider: demo31.ConnectedSerialsProvider | None = None,
        cuda_device_count_provider: demo31.CudaDeviceCountProvider | None = None,
        process_client_factory: demo31.ProcessClientFactory | None = None,
    ) -> None:
        super().__init__(
            args,
            shared_runtime_module=shared_runtime_module,
            shared_runtime_cls=shared_runtime_cls,
            connected_serials_provider=connected_serials_provider,
            cuda_device_count_provider=cuda_device_count_provider,
            process_client_factory=process_client_factory,
        )
        self.contract = build_contract(args, cuda_device_count_provider=cuda_device_count_provider)

    def run(self) -> dict[str, Any]:
        live_validation = validate_live_contract(
            self.args,
            connected_serials_provider=self.connected_serials_provider,
            cuda_device_count_provider=self.cuda_device_count_provider,
        )
        shared = self.shared_runtime_module or demo3_runtime._load_shared_runtime_module()
        shared_profile = demo3_runtime._shared_profile_path(self.args)
        shared_args = build_shared_runtime_args(
            self.args,
            shared_runtime_module=shared,
            live_validation=live_validation,
            shared_profile_path=shared_profile,
        )
        runtime_cls = self.shared_runtime_cls or make_demo32_live_runtime_class(
            shared,
            process_client_factory=self.process_client_factory,
        )
        if self.shared_runtime_cls is None:
            runtime = runtime_cls(
                shared_args,
                demo31_contract=self.contract,
                cotracker_process_config=demo31.build_cotracker_process_config(self.args),
                cotracker_enabled=not bool(self.args.disable_cotracker),
            )
        else:
            runtime = runtime_cls(shared_args)
        exit_code = int(runtime.run())
        shared_payload = demo3_runtime._load_json_if_exists(shared_profile)
        snapshot = runtime.demo31_snapshot() if hasattr(runtime, "demo31_snapshot") else None
        summary = self._build_summary(
            runtime=runtime,
            exit_code=exit_code,
            snapshot=snapshot,
            shared_payload=shared_payload,
        )
        profile = {
            "contract": self.contract,
            "summary": summary,
            "live_validation": live_validation,
            "shared_runtime_profile": None if shared_profile is None else str(shared_profile),
            "shared_runtime_profile_payload": shared_payload,
            "tracker_process_snapshot": snapshot,
            "runtime_note": (
                "Demo 3.2 owns the FFS + point-tracker live orchestration. It reuses "
                "shared camera/FFS/EdgeTAM services but is not a Demo 3.1 preset."
            ),
            "exit_code": exit_code,
        }
        demo31._write_profile(self.args.profile_json_output, profile)
        return profile


def main(
    argv: Sequence[str] | None = None,
    *,
    cuda_device_count_provider: demo31.CudaDeviceCountProvider | None = None,
) -> int:
    parser = build_arg_parser()
    try:
        args = parser.parse_args(argv)
        args = apply_preset_defaults(args, explicit_options=demo3_runtime._explicit_cli_options(argv))
        validate_args(args, require_calibration=False, cuda_device_count_provider=cuda_device_count_provider)
        contract = build_contract(args, cuda_device_count_provider=cuda_device_count_provider)
        if args.dry_run:
            print(format_contract(contract))
            demo31._write_profile(
                args.profile_json_output,
                {"contract": contract, "summary": contract["profile_summary_fields"]},
            )
            return 0
        profile = Demo32Runtime(args, cuda_device_count_provider=cuda_device_count_provider).run()
        print(json.dumps(profile["summary"], indent=2, sort_keys=True))
        return int(profile.get("exit_code", 0))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


__all__ = [
    "DEMO32_RUNTIME_MODULE",
    "DEMO32_RUNTIME_OWNER",
    "Demo32Runtime",
    "PRESET_DEMO32_FFS_TAPNEXTPP",
    "PRESET_DEMO32_FFS_LITETRACKER",
    "apply_preset_defaults",
    "build_arg_parser",
    "build_contract",
    "build_shared_runtime_args",
    "format_contract",
    "main",
    "make_demo32_live_runtime_class",
    "validate_args",
    "validate_live_contract",
]
