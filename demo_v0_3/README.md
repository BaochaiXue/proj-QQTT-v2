# Demo v0.3 Staged Remote FFS

Demo v0.3 is the fixed-replay 100-kit remote FFS benchmark. It does not run
RealSense capture, SAM3.1, EdgeTAM, masks, PCD, Open3D, or Demo 2 rendering.

## Data

Prepare or verify the local 5090 replay folder:

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

The replay folder is binary local data and must not be committed.

## 4090 Server

Use port `7003` for v0.3 staged service tests. Keep existing `7001` and `7002`
services untouched.

```bash
python services/ffs_remote/ffs_depth_staged_server_v03.py \
  --bind tcp://0.0.0.0:7003 \
  --ffs-repo /home/xinjie/Fast-FoundationStereo \
  --ffs-trt-model-dir /home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864 \
  --ffs-trt-batch3-model-dir /home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5_batch3/engines/model_20-30-48_iters_4_res_480x864_batch3 \
  --return depth_u16 \
  --compression lz4 \
  --ffs-mode sequential_batch1 \
  --decode-workers 2 \
  --postprocess-workers 2 \
  --ffs-workers 1 \
  --max-raw-queue 64 \
  --max-decoded-queue 64 \
  --max-postprocess-queue 64 \
  --max-send-queue 64 \
  --warmup 20 \
  --debug \
  --strict-engine-contract
```

Use `--ffs-mode batch3` only after the 4090 100-kit batch3 validate/profile
report passes. The server exits with code `2` if `--ffs-workers` is not exactly
`1`.

## 5090 Client

Run the 15 kit-FPS inflight matrix:

```bash
for N in 1 2 3 6 9 12 16 24 32; do
  conda run --no-capture-output -n demo_2_max \
  python demo_v0_3/staged_remote_ffs_triplet_client.py \
    --mode triplet-replay \
    --replay-dir result/demo_v0_3_ir_triplet_100kits_848x480 \
    --endpoint tcp://192.168.0.162:7003 \
    --capture-kit-fps 15 \
    --warmup-kits 20 \
    --measure-kits 100 \
    --compression lz4 \
    --return-type depth_u16 \
    --max-inflight "$N" \
    --replay-once-measured \
    --drop-stale-replies \
    --save-first-depth-preview \
    --debug
done
```

The client writes:

```text
docs/generated/demo_v03_100kit_remote_<timestamp>.summary.json
docs/generated/demo_v03_100kit_remote_<timestamp>.per_kit.jsonl
```

Stats exclude the 20 warmup kits and include only the 100 measured kits.
