# Demo 3.2 Native RealSense Default

**Goal:** Make Demo 3.2 default to the public `native-realsense` depth backend
while preserving the existing `--depth-backend ir-ffs` opt-in path.

**Scope:** Demo 3.2 wrapper defaults, contract/delegate tests, README wording,
and validation docs. Demo 3.1/3.3 remain unchanged.

## Steps

- [x] Update focused tests so Demo 3.2 default contract/delegate use
      `native-realsense` and internal `depth_source=realsense`.
- [x] Change the Demo 3.2 depth-backend parser default to
      `native-realsense`.
- [x] Update Demo 3.2 docs and hardware validation examples so the default
      examples use native RealSense, with explicit IR-FFS examples as opt-in.
- [x] Run focused runtime validation.
- [x] Run smoke validation.
