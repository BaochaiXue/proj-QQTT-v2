# 2026-05-05 Demo 2 Local FFS Professor Speed Polish

Prepare the local RTX 5090 Laptop Demo 2 path for a low-FPS but usable
professor-facing realtime demo.

Hard constraints:

- main depth source remains FFS-derived depth
- FFS contract remains `20-30-48`, `valid_iters=4`, `848x480 -> 864x480`,
  `builderOptimizationLevel=5`
- mask tracker remains HF EdgeTAM streaming
- EdgeTAM compile mode remains `vision-reduce-overhead`
- no native RealSense depth as the formal output

Optimization scope:

- keep object/controller masked-only PCD, no full-scene PCD
- add a safe local demo preset if useful
- reduce render/UI load without changing FFS depth semantics
- keep startup messaging clear for first-frame SAM3.1 / EdgeTAM warmup
- update docs with the command for tomorrow's local demo

Validation:

- run targeted smoke tests for Demo 2 CLI behavior
- run `python scripts/harness/check_all.py`

Outcome:

- added `--demo-preset local-ffs-professor` for the formal local FFS path
- kept FFS-derived depth and `vision-reduce-overhead` EdgeTAM as hard contracts
- capped default preset PCD points to 20k for the pointcloud demo
- skipped empty controller PCD work in object-only mode
- recorded no-render compute profile in
  `docs/generated/demo2_local_ffs_professor_speed_polish.md`
