# Demo 3.2 Object Erode Zero Default Plan

**Goal:** Make Demo 3.2 and Demo 3.3 default object PCD mask erosion to 0px,
matching the controller default, while preserving explicit CLI override
behavior.

**Scope:** FFS Demo 3.2/3.3 wrapper defaults, focused runtime tests, and Demo
3.2 operator docs. The lower-level masked PCD delegate already supports
per-class erosion and stores effective values in metadata/artifacts.

- [x] Update focused runtime tests so default FFS filtered object erosion is
      0px and explicit `--pcd-mask-erode-pixels` still overrides both classes.
- [x] Change the wrapper default constant from object erosion 3px to 0px.
- [x] Update Demo 3.2 docs to remove the stale 3px object cleanup default.
- [x] Run focused unit tests, Demo 3.2 dry-run, and smoke validation.
