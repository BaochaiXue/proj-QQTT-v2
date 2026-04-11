## Goal

Add a direct 1x3 Rerun viewer layout for the existing multi-frame native-vs-FFS remove-invisible workflow so the three variants can be watched side by side over time.

## Non-Goals

- no new point-cloud generation semantics
- no new artifact family beyond the existing RRD/PLY outputs
- no change to remove-invisible math or point-cloud contracts

## Implementation Plan

1. extend the rerun compare workflow with an optional viewer layout contract
2. build a horizontal triple-view blueprint for:
   - `native`
   - `ffs_remove_1`
   - `ffs_remove_0`
3. activate that blueprint when spawning the viewer and attach it to saved `.rrd` files
4. expose the layout on the existing CLI and document the direct viewer command
5. run the viewer command against the real local cases

## Validation Plan

- keep existing rerun workflow tests green
- run `visual_compare_rerun.py --help`
- run the real viewer command with `--viewer_layout horizontal_triple`
