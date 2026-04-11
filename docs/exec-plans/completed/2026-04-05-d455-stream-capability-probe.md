# 2026-04-05 D455 Stream Capability Probe

## Goal

Build a standalone D455 stream capability probe that measures which stream combinations
are actually stable on this machine for:

- one camera at a time
- all 3 cameras together

The output must support an honest decision about what stream combinations are safe to
consider in the next integration step.

## Non-Goals

- do not modify `record_data.py`
- do not modify `data_process/record_data_align.py`
- do not integrate FFS into the main pipeline
- do not add downstream perception / rendering / simulation code
- do not claim support without a measured hardware result

## Current Validated Baseline

- external Fast-FoundationStereo checkout exists
- checkpoint `23-36-37` exists locally
- official FFS demo works
- one D455 single-camera proof-of-life already exists for serial `239222303506`
- saved D455 IR pair already runs through FFS and converts to color-aligned compatible depth

## Local Path Assumptions

- QQTT repo root: `C:\Users\zhang\proj-QQTT`
- External FFS repo: `C:\Users\zhang\external\Fast-FoundationStereo`
- Probe output root: `data/ffs_proof_of_life/d455_stream_probe/`
- Generated docs:
  - `docs/generated/d455_stream_probe_results.json`
  - `docs/generated/d455_stream_probe_results.md`

## Probe Matrix

Topologies:

- single camera:
  - `239222300433`
  - `239222300781`
  - `239222303506`
- three cameras together in stable order:
  - `239222300433`
  - `239222300781`
  - `239222303506`

Primary profiles:

- `848x480@30`
- fallback `640x480@30` only when needed

Stream-set names:

- `depth`
- `color`
- `ir_left`
- `ir_right`
- `ir_pair`
- `rgbd`
- `rgb_ir_pair`
- `depth_ir_pair`
- `rgbd_ir_pair`

Emitter policy:

- first pass:
  - `auto` for non-IR cases
  - `on` for IR-containing cases
- second pass:
  - `off` only for IR-containing cases that already succeeded

## Measurement Methodology

For each case:

- open raw `pyrealsense2` pipelines directly
- measure start success/failure
- record time to first frame
- run explicit warmup and measurement windows
- count frames per stream
- estimate observed fps per stream
- check timestamp monotonicity
- detect mid-run stalls / timeouts
- record exact error text on failure
- save one sample frame per requested stream for successful cases

## Output Artifacts

- `data/ffs_proof_of_life/d455_stream_probe/runs/<timestamp>/results.json`
- `data/ffs_proof_of_life/d455_stream_probe/runs/<timestamp>/summary.md`
- `data/ffs_proof_of_life/d455_stream_probe/latest/...`
- `docs/generated/d455_stream_probe_results.json`
- `docs/generated/d455_stream_probe_results.md`

## Validation Plan

Software-only:

- `pytest tests/test_d455_probe_matrix_builder.py`
- `pytest tests/test_d455_probe_result_schema.py`

Hardware:

- run single-camera probe for each serial
- run one three-camera probe matrix
- verify sample frames and failure logs are written
- render JSON + Markdown reports

## Risks

- some multi-stream combinations may start but fail stability thresholds
- 3-camera runs may expose USB bandwidth or topology bottlenecks that do not appear in single-camera runs
- emitter on/off may change behavior for IR cases
- RealSense SDK-reported stream availability may still differ from observed stable delivery

## Acceptance Criteria

- standalone probe script exists
- report renderer exists
- one-camera and three-camera topologies are both measured
- required stream sets are covered
- results are saved in JSON and Markdown
- successful cases save sample frames
- software-only matrix/schema tests exist
- main pipeline files remain untouched
- final recommendation is grounded in measured evidence

## Completion Checklist

- [ ] add probe script
- [ ] add report renderer
- [ ] add matrix/schema tests
- [ ] run software-only tests
- [ ] run single-camera probe for all serials
- [ ] run three-camera probe matrix
- [ ] write generated JSON / Markdown summaries
- [ ] move this plan to `docs/exec-plans/completed/`

## Progress Log

- 2026-04-05: created active execution plan for D455 stream capability probing
