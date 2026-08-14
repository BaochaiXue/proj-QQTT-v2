# Phystwin stage-specific window CLI proof — 2026-07-10

## Scope

Verified the cross-repository CLI contract for `batch_size`, `segment_len`,
and `segment_stride` overrides under Stage 1, Stage 2, and Train. No camera or
GPU workload was started.

## External wrapper

Checkout: `/home/xinjie/Phystwin_shen`, branch `online` (starting HEAD
`5dc5f40`, validated/pushed commit `0441dc6`).

Command:

```bash
conda run -n demo_2_max --no-capture-output \
  python -m unittest discover -s tests -p 'test_*.py' -v
```

Outcome: 4 tests passed, including typed stage overrides and
common-then-stage precedence.

Parser/child-command probe:

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/run_online_full_pipeline.py \
  --config configs/online_full_pipeline.yaml \
  --online_dir /tmp/phystwin-stage-window-contract/online_data \
  --cuda_visible_devices 1 --dry_run \
  --skip_cma_viewer --skip_train_viewer \
  --stage1_batch_size 2 --stage1_segment_len 10 \
  --stage1_segment_stride 10 \
  --stage2_batch_size 3 --stage2_segment_len 20 \
  --stage2_segment_stride 20 \
  --train_batch_size 5 --train_segment_len 30 \
  --train_segment_stride 30
```

Outcome: parser exit 0. Printed child commands contained exactly:

- Stage 1: `--batch_size 2 --segment_len 10 --segment_stride 10`
- Stage 2: `--batch_size 3 --segment_len 20 --segment_stride 20`
- Train: `--batch_size 5 --segment_len 30 --segment_stride 30`

## Demo v6.2 bridge

Command:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v6_2/main.py --dry-run
```

Outcome: exit 0. The generated supervisor command omitted absent common
window flags and included Stage 1 `2/10/10` plus Train `5/30/30` overrides.
Disabled Stage 2 emitted no window overrides.

Focused validation:

```bash
conda run -n demo_2_max --no-capture-output \
  python -m pytest tests/test_demo_v6_2_downstream.py -q
```

Outcome: 48 tests and 17 subtests passed.

Repository validation:

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/validation/run.py --profile smoke
```

Outcome: smoke passed; 243 unit tests passed with all guards and help probes.
