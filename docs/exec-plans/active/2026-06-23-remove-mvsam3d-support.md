# Remove Multi-View SAM Shape-Prior Support

## Goal

Remove the obsolete multi-view SAM shape-prior code paths and references while
keeping SAM3D/SAM3D-objects single-view shape-prior support for Demo 3.2 and
offline SAM3D use.

## Steps

1. Delete the old multi-view-only modules under `data_process_sam3d`.
2. Remove old root fallbacks and help text from SAM3D-only code.
3. Simplify shape-prior sampling so there is only one SAM3D path.
4. Update tests/docs to stop advertising removed backend support.
5. Verify targeted tests, smoke validation, and repository search for removed terms.
