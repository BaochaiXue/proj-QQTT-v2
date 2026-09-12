# Demo v7 concurrency review: fact check and direct corrections

## Requirement

- Problem: verify the review of commit 35fbf5f against the actual implementation
  and correct confirmed lifecycle/display defects without speculative recovery.
- Required behavior: link-death notifications cannot be lost; accepted commands
  own real work; artifacts merge consistently; terminal errors stop production;
  successful finalization implies producers/writers have stopped; live preview
  retains bounded latest frames before Qt queuing.
- Inputs: current main (ff38541, only CI/test isolation changed since 35fbf5f),
  recorded cases, deterministic thread/socket failure injection.
- Outputs: narrowly scoped fixes, regression evidence, and a claim-by-claim
  fact-check under docs/generated/.
- State changes: existing session/link/worker ownership and GUI preview only.
- Invalid cases: rejected work gets a negative ACK; failed accepted work emits
  the existing error model; incomplete drain cannot become FINISHED.
- Constraints: preserve model algorithms, 5 FPS timeline, lossless products,
  and existing optional Gaussian/recorder error policy. No new fallback path,
  generic retry layer, speculative writer queue, or unmeasured algorithm change.
- Unknowns: severity/throughput impact requires measurements; physical USB
  disconnection cannot be inferred from simulated faults.

## Minimal design and validation

1. Reproduce correctness claims with deterministic synchronization. Repair
   reconnect ownership with a per-link dirty notification, merge/snapshot
   artifacts under one lock, and reserve actual Gaussian work before ACK.
2. Use existing teardown/process-group ownership for terminal link failure;
   check actual worker termination before successful finalization. Correct
   recorder write/metadata integrity where verified.
3. Bound GUI ingress before queued signals, avoid stale socket batches, and
   move channel eligibility ahead of preview derivation. Keep optimizations
   whose necessity is not established as measured follow-up findings.
4. Run Demo v7 regressions, repo smoke (exhaustive if needed), focused socket/Qt
   stalls, and real GPU fake-live control flow. Record commands and limits.
5. Commit/push validated changes to origin main, with an explicit fact-check
   distinguishing source-level defects, reproduced outcomes, and unmeasured
   performance hypotheses.

## Initial findings

- Lost notification during reconnect installation is real; a failed HELLO is
  currently swallowed and the dead client can remain installed.
- Artifact merge and HELLO snapshot lack synchronization.
- Regen ACK precedes admission. TripoSplat pipe write failure already emits an
  error (so that failure is not wholly silent); its false return is ignored.
  Mesh-surface admission also races because busy is set in its worker.
- Qt queues JPEG payloads before ImageView coalescing. 55 Hz is a configured
  ceiling, not a measurement of the default 5 FPS fake-live path.
