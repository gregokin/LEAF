# Additional independent evidence for revision 1 checkpoint finding

`probe_checkpoint_parent_01.py` runs the genuine N15 native factory, one daily
operation and one LEAF-proposed withdrawal, stages/publishes biology and saves
the biological checkpoint. It then makes an additional genuine zero-time
withdrawal in SYRUP, which has no corresponding LEAF biological update, and
supplies that later state to the biological loader.

The revision 1 loader accepts it: saved request sequence 1 versus supplied
sequence 2; saved physical transpiration 5.189791448709424e-5 m3 versus supplied
5.339791448709424e-5 m3. This is an unpaired physical state, not an ordinary
post-T phase close. The exact pairing must be saved separately from the last
consumed T cursor so legitimate closed-boundary saves are supported explicitly.

Evidence: `probe_checkpoint_parent_01.command.json`, launch/exit records and
separate stdout/stderr logs. The specification assertion deliberately fails
with exit 1. The initial probe-configuration setup had a missing Python `json`
import; it failed before any model execution and was corrected before the
recorded probe launch. This is a verification setup error, not a model defect.
