# Elmer case template (Stage 4)

`saw_idt.sif.template` defines the coupled piezoelectric problem that would
replace Stage 1's assumed voltage-to-pressure calibration with a solved one.

## What it removes

Stage 1 uses `PRESSURE_PER_VOLT_ASSUMPTION` — a linear device constant anchored
at 0.45 MPa for 15 Vpp — because converting IDT drive voltage into substrate
displacement needs the electromechanical solution. That assumption is the single
largest unsourced input in the sorter, and this back-end is how it gets removed.

## Bridge contract

1. biosim-lab meshes the substrate + IDT with `core.geometry.idt_electrodes_2d`
   (extruded to 3-D) and converts it with `ElmerGrid`.
2. `saw_idt.sif` is rendered with the drive frequency, voltage and the rotated
   material tensors.
3. `ElmerSolver` runs; `SaveLine` writes `surface_displacement.dat` along the
   substrate surface.
4. `solver_elmer.surface_displacement_to_bc()` turns that into the
   prescribed-velocity boundary condition that
   `saw_sorter.fem_model.SAWFieldModel.boundary_conditions()` already accepts —
   so switching from the assumed calibration to the solved one changes one line
   in the sorter and nothing else.

## The rotation is the fiddly part

The published LiNbO3 constants are in the crystal frame. 128° YX is a rotated
Y-cut, so `c^E`, `e` and `eps^S` must all be rotated by the Bond transformation
before they go into the `.sif`. The template leaves `${C_ROTATED}` etc. as
substitution points precisely so the rotation happens once, in Python, where it
can be unit-tested against the published free-surface SAW velocity of
3979–3992 m/s (`materials.LINBO3_128YX.saw_velocity`).
