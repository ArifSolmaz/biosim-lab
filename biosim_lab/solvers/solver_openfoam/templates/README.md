# OpenFOAM case templates (Stage 4)

These files define the *interface contract* between biosim-lab and OpenFOAM.
They are shipped so that the boundary between the two is explicit and
reviewable; the Python bridge that fills them in is not implemented yet
(`OpenFOAMSolver.setup()` raises `NotImplementedError` with this path in the
message).

| File | Role |
|------|------|
| `acousticRadiationForce.H` / `.C` | DPMFoam particle-force model that reads the complex acoustic pressure field written by biosim-lab, forms the Gor'kov potential on the mesh and applies `-grad(U)` to each parcel. |
| `controlDict.template` | Run control plus the `fieldAverage` function object that extracts the steady (second-order) streaming field, and a placeholder for the Nyborg boundary-layer source term. |
| `kinematicCloudProperties.template` | Wires the custom force into DPMFoam's `particleForces` list and injects a log-normal cell-size distribution. |

## Intended workflow

1. biosim-lab writes the Gmsh mesh, runs `gmshToFoam`.
2. The Stage 1 Helmholtz solution is exported as `0/pAcousticRe` and
   `0/pAcousticIm` (`volScalarField`).
3. Templates are rendered with the material and drive parameters.
4. `DPMFoam` runs; `fieldAverage` produces `UMean`, the streaming field.
5. `fluidfoam` reads `UMean` back; biosim-lab adds it to the Lagrangian
   tracker's fluid velocity as `u_streaming`.

## Validation gate before trusting results

Compare against the analytic Rayleigh streaming solution for a 2-D channel:
the slip velocity outside the acoustic boundary layer is
`u_slip = -(3/8) u1^2 / c`, giving a bulk streaming magnitude of
`(3/16) u1^2 / c` (Rayleigh 1884, doi:10.1098/rstl.1884.0002; Bruus,
doi:10.1039/c1lc20770a). `tests/test_solver_plugins.py` contains that check,
skipped when the back-end is absent.

## Build

```bash
# inside the biosim-lab-openfoam image
cp acousticRadiationForce.[CH] $FOAM_RUN/../applications/solvers/lagrangian/DPMFoam/
# add acousticRadiationForce.C to Make/files, then
wmake
```
