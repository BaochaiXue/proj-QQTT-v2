# Demo v6.2 live CUDA graph safety

## Problem

The strict Demo v6.2 camera runtime runs EdgeTAM segmentation and TAPNext++
tracking in separate Python threads on the same CUDA device. EdgeTAM uses
`torch.compile(mode="reduce-overhead")`, which records its CUDA graph on the
second model call. A live-camera run failed on its first tracking frame with
PyTorch's
`Offset increment outside graph capture encountered unexpectedly` error.

A minimal two-thread CUDA probe reproduces the exact exception when one thread
performs graph capture while the other executes a CUDA RNG operation. In the
failed run, the tracker fatal event occurred 215 ms after frame 0, matching the
5 FPS live sampler's next-frame tick. Live waited only for frame-0 segmentation,
so EdgeTAM call 2 recorded its graph while TAPNext++ lazily constructed its CUDA
model. Fake-live already avoids this race by waiting for the complete frame-0
PCD/tracker pair before releasing frame 1.

## Required behavior

- Live and fake-live strict runtimes must use the same frame-0 startup
  handshake: frame 1 cannot be released until frame 0 has complete PCD and
  tracker results.
- TAPNext++ failures must remain fatal; the runtime must not retry, skip a
  frame, or publish partial tracking data.
- The 5 FPS output contract, EdgeTAM compile mode, and model/device assignments
  must remain unchanged.

## Changes

- Generalize the existing lossless replay first-pair gate so it also covers
  live input.
- Start the live 5 FPS sampler only after that gate opens.
- Add regression tests for the live first-pair wait and its ordering before the
  sampler starts.

## Validation

1. Run the focused live-startup regression tests.
2. Run a one-chunk live-camera pipeline with downstream disabled and confirm
   the tracker publishes a complete chunk without the CUDA generator error.
3. Inspect the resulting manifest and run summary.
4. Run the repository smoke profile.

## Result

- The focused live-startup and cleanup suites pass 10 tests.
- A one-chunk fake-live end-to-end run completed 35 frames, published a
  `finished` manifest, and reported one `normal` tracking chunk with no CUDA
  generator failure or process leak.
- The controlled live attempt reached the first-frame semantic gate but the
  unattended scene contained no detectable `hand`; it failed explicitly before
  TAPNext++ initialization. A live first-pair hardware rerun remains required
  with the controller and object visible and stationary.
- The repository smoke profile passes all 231 tests.
