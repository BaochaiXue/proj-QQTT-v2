# Demo 3.2 Lossless Runtime Backpressure

## Goal

Make Demo 3.2 strict lossless tracking run full fake-live cases with the normal
15-frame backlog. Runtime tracker slowdowns should apply backpressure to the
faster PCD side instead of failing the pairer when PCD reaches one-sided
pending depth.

## Root Cause

After startup warmup, the PCD worker can build and submit filtered point-cloud
results faster than TAPNext++ produces marker packets. The same-seq pairer is a
bounded reorder buffer, but workers currently submit into it until it exceeds
the configured backlog and raises a fatal `LosslessPipelineError`.

## Implementation

1. Add pairer-side condition-based capacity waiting for the fast side.
2. Add queue-side condition-based capacity waiting for lossless frame/mask queues.
3. Have PCD and tracker workers wait for their side's capacity before entering
   the outer pairer publish lock and submitting results.
4. Have capture/segmentation workers wait for downstream lossless queue capacity
   instead of treating normal downstream backpressure as fatal backlog.
5. Keep fatal backlog errors as a defense for direct misuse or logic bugs that
   bypass the worker backpressure.
6. Validate with red/green unit tests, targeted Demo 3.x tests, and a default
   backlog fake-live tracking run.
