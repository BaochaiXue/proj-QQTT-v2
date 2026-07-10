# Promote the single-camera line to `main`

## Goal

Preserve the former `main` tip as `multiple-camera`, then make the current
single-camera development line the repository's canonical `main` branch.

## Preflight

- Former `origin/main`: `42076d5f1322865ce1d013acb214156354d720d1`.
- Current `origin/single-camera`: `ac16a13a73d91fc6b905e9357810585b895c54cf`.
- The branches diverged at `c7645b1032a819528feb16c2cc3d1d6f412dc946`;
  the former main has one unique commit and single-camera has 300.
- `origin/multiple-camera` does not exist before this migration.
- The Demo GPU-routing configuration is included in the Demo runtime commit.
- Local `simple-knn` build directories are generated artifacts and excluded
  from both commits.

## Migration

1. Update branch-policy documentation and its deterministic scope guard.
2. Validate and commit the Demo GPU-routing change and branch-policy migration
   files as separate commits.
3. Create `origin/multiple-camera` at the exact former `origin/main` tip and
   verify the remote hash.
4. Update `origin/main` to the promoted single-camera commit with an explicit
   force-with-lease against the audited former-main hash.
5. Verify both remote hashes, then remove the obsolete remote
   `single-camera` ref.
6. Rename local branches to match their new roles and verify upstreams.

## Validation

- Run the scope guard directly.
- Run the repository smoke validation profile.
- Confirm `origin/main` and local `main` resolve to the promoted commit.
- Confirm `origin/multiple-camera` and local `multiple-camera` resolve to the
  former-main commit.
- Confirm the remote default branch remains `main`.
