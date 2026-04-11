# both_eval 30-Frame Validation

## Goal

Check whether this machine can simultaneously record:

- native RealSense `depth`
- `ir_left`
- `ir_right`

for the 3-camera D455 setup, using the repo's `both_eval` capture mode.

## Current Hardware Context

- date: `2026-04-11`
- serials:
  - `239222300433`
  - `239222300781`
  - `239222303506`
- profile:
  - `848x480 @ 30 FPS`
  - `emitter=on`

## 1. Current Repo Policy Result

Preflight evaluation:

```text
capture_mode = both_eval
topology = three_camera
stream_set = rgbd_ir_pair
probe_support = False
policy = block_if_probe_fails
allowed_to_record = False
```

Official recording command:

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe record_data.py --case_name tmp_both_eval_30f_policy_check --capture_mode both_eval --emitter on --max_frames 30 --disable-keyboard-listener
```

Observed result:

- `record_data.py` enumerated all 3 cameras
- `Camera system is ready.`
- recording was then blocked by preflight after camera discovery
- no experimental raw case was kept

Blocking reason:

- current repo policy still trusts `docs/generated/d455_stream_probe_results.*`
- that probe marked 3-camera `rgbd_ir_pair` unsupported / unstable on this machine

## 2. Direct Runtime Experiment (Bypassing Preflight)

Because the user asked whether simultaneous capture can actually work now, I ran a direct `CameraSystem(capture_mode='both_eval')` experiment outside `record_data.py` policy gating.

Initialization probe result:

- success
- all 3 cameras produced synchronized payloads containing:
  - `color`
  - `depth`
  - `ir_left`
  - `ir_right`

Short recording experiment:

- mode: direct `CameraSystem(..., capture_mode='both_eval', emitter='on')`
- target: `max_frames=30`
- raw output path used during the experiment:
  - `data_collect/tmp_both_eval_30f_direct/`

Observed result:

- success
- the recorder exited normally
- all 3 cameras reached `30` saved frames for every requested stream

Saved-frame counts before cleanup:

```json
{
  "streams_present": [
    "color",
    "depth",
    "ir_left",
    "ir_right"
  ],
  "counts": {
    "color": {
      "0": 30,
      "1": 30,
      "2": 30
    },
    "depth": {
      "0": 30,
      "1": 30,
      "2": 30
    },
    "ir_left": {
      "0": 30,
      "1": 30,
      "2": 30
    },
    "ir_right": {
      "0": 30,
      "1": 30,
      "2": 30
    }
  }
}
```

## Conclusion

- **Yes, this machine was able to record native depth and IR images simultaneously for a 30-frame 3-camera short run.**
- **No, the current repo policy does not allow that through `record_data.py --capture_mode both_eval` yet**, because the latest persisted D455 stream probe still marks the profile unsupported.

Interpretation:

- this experiment proves short-run feasibility for `30` frames on the current hardware state
- it does **not** prove long-run stability
- the discrepancy is between:
  - old probe-based policy: still blocks
  - current short runtime experiment: succeeded

Most likely reason:

- the saved probe result is now stale or stricter than the current short-run requirement
- alternatively, the profile may still be fragile over longer durations even though a 30-frame burst succeeds

## Cleanup

Experimental raw data was removed after validation:

- `data_collect/tmp_both_eval_30f_direct/`
- no retained `tmp_both_eval_30f_policy_check/` directory remained
