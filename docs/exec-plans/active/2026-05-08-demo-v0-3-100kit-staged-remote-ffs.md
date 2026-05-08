# Demo v0.3 100-Kit Staged Remote FFS

## Status

Active. Demo v0.3 is the next remote FFS track. Do not extend Demo v0.2 except
to read its existing replay folder as source data or historical reference.

## Goal

Build a fixed-replay benchmark for the RTX 5090 laptop to RTX 4090 remote FFS
path:

```text
100 measured IR triplet kits
20 warmup kits excluded from stats
15 kit-FPS capture cadence
avg / min / max / p50 / p90 / p95 / p99 over measured kits only
```

Each kit contains synchronized cam0/cam1/cam2 IR pairs:

```text
cam0 left/right + cam1 left/right + cam2 left/right
```

## Scope

- Add Demo v0.3 helper code under `scripts/demo_v0_3/` and `demo_v0_3/`.
- Prepare `result/demo_v0_3_ir_triplet_100kits_848x480` from the existing
  v0.2 real IR replay folder without committing binary data.
- Keep Demo v0.3 managed by harness engineering docs, generated reports, and
  deterministic smoke tests.
- Add warmup/measured split to v0.3 client and local 4090 profile scripts.
- Use 7003 for staged v0.3 service tests; do not disturb 7001/7002.

## Non-Goals

- No live RealSense capture in the first v0.3 profiling pass.
- No SAM3.1, EdgeTAM, masks, PCD, Open3D, or demo rendering.
- No single-triplet benchmark claims.
- No binary replay data in Git.

## 5090 Data Preparation

```bash
conda run --no-capture-output -n demo_2_max \
python scripts/demo_v0_3/prepare_ir_triplet_100kits.py \
  --src-replay-dir result/demo_v0_2_data_ir_triplet_replay_848x480_still_object_round8 \
  --out-replay-dir result/demo_v0_3_ir_triplet_100kits_848x480 \
  --num-kits 100 \
  --camera-count 3 \
  --width 848 \
  --height 480 \
  --capture-kit-fps 15 \
  --allow-cycle-if-needed \
  --write-manifest \
  --debug
```

Output files:

```text
result/demo_v0_3_ir_triplet_100kits_848x480/metadata.json
result/demo_v0_3_ir_triplet_100kits_848x480/manifest_v03_100kits.json
result/demo_v0_3_ir_triplet_100kits_848x480/kits.jsonl
```

## Profiling Contract

- `warmup_kits=20`
- `measure_kits=100`
- `capture_kit_fps=15`
- `kit_period_ms=66.6667`
- Warmup requests may reuse the same 100-kit folder but must not enter
  latency, throughput, or stage summaries.

## Reporting

Primary report:

```text
docs/generated/demo_v03_100kit_replay_validation.md
```

Expected machine-readable outputs:

```text
docs/generated/demo_v03_100kit_profile_<timestamp>.summary.json
docs/generated/demo_v03_100kit_profile_<timestamp>.per_kit.jsonl
```

## Validation

```bash
python -m py_compile scripts/demo_v0_3/prepare_ir_triplet_100kits.py
conda run --no-capture-output -n demo_2_max python -m unittest -v tests.test_demo_v03_prepare_ir_triplets_smoke
conda run --no-capture-output -n demo_2_max python scripts/harness/check_all.py
```
