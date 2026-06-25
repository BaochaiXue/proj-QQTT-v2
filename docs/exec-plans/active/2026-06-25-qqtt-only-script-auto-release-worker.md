# QQTT-only Script Auto-release Shape Worker

Goal: update `scripts/run_single_proj_qqtt_only_fake_live.sh` so the managed
SAM3D shape-prior worker is stopped automatically after Demo v4 has consumed a
ready shape prior, while Demo v4 continues producing chunks/final_data.

Scope:
- Keep the change in the wrapper script and tests.
- Do not modify Demo v4 runtime semantics.
- Do not start Demo v5 or `realtime_phystwin` optimization.
- Only stop workers started by this wrapper. Existing external workers are left
  alone.

Approach:
1. Start Demo v4 as a subprocess instead of `exec`.
2. Run a background monitor while Demo v4 is alive.
3. Stop the managed worker once both are true:
   - a `shape_prior/points.npz` exists under the selected Demo v4 output root;
   - at least one published chunk manifest reports
     `shape_prior_complete` or `shape_prior_target_counts_met`.
4. Keep the existing `EXIT` cleanup as a fallback for failures or early exits.

Validation:
- `bash -n scripts/run_single_proj_qqtt_only_fake_live.sh`
- `python -m unittest tests.test_qqtt_only_script`
- dry-run wrapper invocation with worker management disabled
- smoke validation profile
