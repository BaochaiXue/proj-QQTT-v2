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

## 3. Longer-Duration Stability Probe

Targeted long-duration probe:

- topology: `three_camera`
- stream set: `rgbd_ir_pair`
- profile: `848x480@30`
- emitter: `on`
- warmup: `2.0s`
- duration: `30.0s`

Observed result:

- the case started successfully
- all requested streams were delivered on all 3 cameras
- the case still failed the probe's stability thresholds
- failure type:
  - `StabilityThresholdNotMet`
- updated long-duration note:
  - the refreshed probe case is now recorded in `docs/generated/d455_stream_probe_results.*`

Key failure details:

- camera `239222300433`
  - max timestamp gap reached about `603 ms`
- camera `239222300781`
  - max timestamp gap reached about `889 ms`
- camera `239222303506`
  - max timestamp gap reached about `442 ms`

Those gaps are far above the probe's allowed `max_timestamp_gap_factor`, so the long-duration run remains unstable even though the short 30-frame burst completed.

## Final Conclusion

- **Yes, this machine can complete a short 30-frame 3-camera `both_eval` burst and save `color + depth + ir_left + ir_right`.**
- **No, this machine still does not pass a longer 30-second `rgbd_ir_pair` stability probe.**
- **Therefore the default repo strategy should remain probe-gated and blocked for `record_data.py --capture_mode both_eval` on this machine/profile.**

Interpretation:

- short-burst feasibility is now proven
- long-duration stability is still not proven
- for repo policy, long-duration stability remains the deciding requirement
- the previous conservative `both_eval` gate remains the correct default behavior after this revalidation

## Cleanup

Experimental raw data was removed after validation:

- `data_collect/tmp_both_eval_30f_direct/`
- no retained `tmp_both_eval_30f_policy_check/` directory remained
- temporary targeted probe artifacts under `%TEMP%\\qqtt_both_eval_long_probe\\` were removed after the case result was merged into docs
