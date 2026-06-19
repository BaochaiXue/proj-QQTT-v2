# Demo 3.2 Lossless Fake-Live Warmup Gate

## Goal

Prevent Demo 3.2 fake-live strict lossless mode from counting startup warmup
time against the same-seq PCD/tracker backlog. Startup includes first-frame
segmentation, TAPNext++ adapter loading, initial query selection, first PCD
build, and first same-seq pair publication.

## Root Cause

The replay capture worker publishes seq 0, waits only until the first frame is
segmented, then resumes the fake-live replay clock. While TAPNext++ is still
loading/initializing, PCD results can advance into the pairer and exceed the
lossless backlog while the pairer still waits for seq 0 tracker output.

## Implementation

1. Add a lossless first-pair event that is reset with the lossless queues and
   set when seq 0 is published as a strict pair.
2. For replay capture workers in lossless tracking mode, wait for that first
   pair event before starting the replay clock after first-frame segmentation.
3. Keep runtime seq numbering from 0/1 as-is; skipped fake-live source frames
   remain warmup frames and are not part of strict same-seq accounting.
4. Validate with the new unit regression and the targeted Demo 3.x unit suite.

