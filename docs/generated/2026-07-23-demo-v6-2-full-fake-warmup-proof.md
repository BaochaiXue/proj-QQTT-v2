# Demo v6.2 Full Fake-Live Warm-up Proof (2026-07-23)

## Scope

This is a complete formal Demo v6.2 fake-live run, not a bounded upstream
smoke run. It used the default `phystwin_shen` downstream, replayed the source
until natural completion, and waited for Stage 1, online training, the combined
viewer, and the Phystwin supervisor to exit.

- Repository commit: `6b2baa4dfe145ad91e8441d4a5eb40cb5c1b177c`
- Conda environment: `demo_2_max`
- Input case: `data_collect/sloth_new_20260705_230611`
- Run output: `/tmp/demo_v6_2_full_fake_20260723`
- Persistent shape cache: `/home/xinjie/qqtt_shape_prior_cache`

## Exact command

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v6_2/main.py \
  --input-source fake-live \
  --base-path /tmp/demo_v6_2_full_fake_20260723 \
  --no-warmup-rgb-preview
```

No `--max-chunks` or downstream override was supplied.

## Warm-up result

The authoritative source is
`capture/shape_prior_profile.json` under the run output.

| Measurement | Time |
| --- | ---: |
| `warmup_total_ms` | 20,461.293 ms |
| Runtime start to shape-prior ready | 20,440.676 ms |
| Shape-prior ready to warm-up gate open | 20.617 ms |
| Runtime start to frame-0 receive | 455.298 ms |
| Frame-0 receive to mask ready | 14,388.148 ms |
| Frame-0 mask step | 2,353.180 ms |
| EdgeTAM frame-0 model forward | 2,335.424 ms |
| Frame-0 PCD step | 411.084 ms |
| Runtime start to shape-prior submit | 15,254.860 ms |
| Shape-prior request total | 5,185.822 ms |
| EdgeTAM dummy precompile | 4,781.081 ms |

The shape-prior request decomposed as follows:

| Stage | Time |
| --- | ---: |
| Case write | 245.885 ms |
| Cache-hit mesh materialization (`generate`) | 2.103 ms |
| Align | 3,714.096 ms |
| Sample | 853.199 ms |
| Worker reap | 368.334 ms |
| Result finalize | 1.430 ms |

The profile reported `shape_prior_status=ready`,
`shape_prior_cache_status=hit`, and the critical path ended in:

```text
... -> align -> sample -> worker_reap -> result_finalize
```

## End-to-end result

- Top-level command return code: `0`
- Total orchestrator wall time: `204.406 s`
- Main data-processing return code before stop: `0`
- Main data-processing stop reason: `main_data_processing_completed`
- Published chunks: `169`
- Published formal frames: `845`
- Manifest terminal state: `finished`
- Latest committed chunk: `168`
- Skipped online publishes: `0`
- Phystwin launch trigger: `shape_prior_points_ready`
- Phystwin supervisor return code: `0`
- Stage 1 completed both configured CMA iterations.
- Online train observed the finished stream and stopped after iteration 13.
- The saved Phystwin process group had no remaining processes after exit.
- The combined viewer port `8765` was no longer listening after exit.

Tracking quality telemetry reported one `normal` chunk, 168 `degraded`
chunks, and zero `invalid` chunks. This did not break publication or the
downstream run, but it is a distinct tracking-quality observation and should
not be presented as all chunks being `normal`.

## Artifact references

- `/tmp/demo_v6_2_full_fake_20260723/capture/shape_prior_profile.json`
- `/tmp/demo_v6_2_full_fake_20260723/online_data/manifest.json`
- `/tmp/demo_v6_2_full_fake_20260723/run_summary.json`
- `/tmp/demo_v6_2_full_fake_20260723/pipeline_status.jsonl`
- `/tmp/demo_v6_2_full_fake_20260723/phystwin_shen/online_full_pipeline.log`
