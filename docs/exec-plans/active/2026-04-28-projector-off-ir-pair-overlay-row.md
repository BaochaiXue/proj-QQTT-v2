# Projector-Off IR Pair Overlay Row

Add a diagnostic variant of the enhanced PhysTwin removed-points overlay for
projector-off IR captures.

## Plan

1. Keep the existing native-depth `5x3` board as the default behavior.
2. Add an `ir_pair` row mode that replaces the native-depth row with two rows:
   - IR left
   - IR right
3. Record the selected row mode in the board filename, summary, and render
   contract so the output is self-describing.
4. Add smoke coverage for the new `6x3` IR-pair board mode.
5. Regenerate the projector-off round 1 panel with the IR-left / IR-right rows.

## Notes

- This is an experiment visualization change only.
- Aligned case data and formal recording/alignment contracts are unchanged.
