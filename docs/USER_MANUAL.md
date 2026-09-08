# biosim-lab — User Manual

Version 0.1.0 · MIT licence

> **Language note.** This manual is in English, matching the explainer PDF. The
> README and the reference docs (`physics.md`, `validation.md`,
> `instruments.md`) are Turkish-first. Ask if you would like a Turkish edition
> of this manual too.

---

## Table of contents

1. [What this is, in one page](#1-what-this-is-in-one-page)
2. [Three ways to use it](#2-three-ways-to-use-it)
3. [Installing locally](#3-installing-locally)
4. [The web app, page by page](#4-the-web-app-page-by-page)
5. [The command line](#5-the-command-line)
6. [The Python API](#6-the-python-api)
7. [Configuration file reference](#7-configuration-file-reference)
8. [Using your own data](#8-using-your-own-data)
9. [Publishing your own copy on Streamlit](#9-publishing-your-own-copy-on-streamlit)
10. [Docker](#10-docker)
11. [Reading results honestly](#11-reading-results-honestly)
12. [Troubleshooting](#12-troubleshooting)
13. [Extending it](#13-extending-it)

---

## 1. What this is, in one page

biosim-lab simulates what four kinds of laboratory instrument measure, using
only open-source Python. It also runs the same analyses on real instrument data.

| Instrument | Commercial equivalent | Answers | Status |
|---|---|---|---|
| `saw_sorter` | acoustic cell separators | Can I pull tumour cells out of blood, and how pure is the result? | complete |
| `impedance_rtca` | xCELLigence RTCA | How fast are the cells growing, and what drug dose kills half of them? | minimal but working |
| `cell_counter` | Countess, Cellometer | How many cells per mL, and what fraction are alive? | skeleton + demo |
| `cell_tracker` | Incucyte | How fast do they crawl, and do they move in a direction? | skeleton + demo |

Everything shares one core: a validated configuration, a solver, a result
container, and a plotting layer. Adding a fifth instrument requires no change to
that core — see [§13](#13-extending-it).

**If you only read one other thing**, read
[`docs/explainer/biosim-lab-explained.pdf`](explainer/) — 29 pages on the
biology, physics and computation, written for non-experts, of which about a
fifth is what the model gets wrong.

---

## 2. Three ways to use it

| Route | Best for | Needs |
|---|---|---|
| **Web app** | exploring, teaching, showing someone | a browser (hosted) or `pip install -r requirements.txt` |
| **Command line** | reproducible runs, batch sweeps, saving results | a local install |
| **Python API** | your own analysis, notebooks, new instruments | a local install |

They are the same code. The web app calls exactly the functions the command line
calls; nothing on screen is pre-computed.

---

## 3. Installing locally

Requires **Python 3.11 or newer**.

```bash
git clone https://github.com/<your-account>/biosim-lab.git
cd biosim-lab
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[mesh]"
biosim doctor                      # confirms what works
```

### Which install do I want?

| Command | Gets you |
|---|---|
| `pip install -e .` | everything except Gmsh meshing |
| `pip install -e ".[mesh]"` | **recommended** — adds Gmsh + meshio, needed only for PDMS wall layers |
| `pip install -r requirements.txt` | the slim web-app set; no PyVista, Panel, Gmsh or NetCDF |
| `pip install -e ".[imaging]"` | Napari viewer and `btrack` lineage trees |
| `pip install -e ".[segmentation]"` | Cellpose / StarDist segmentation back-ends |
| `pip install -e ".[dev]"` | pytest, ruff, mypy |
| `pip install -e ".[all]"` | everything above — ~200 packages, pulls PyTorch |

### What cannot go on the hosted app, and why

The web app deliberately installs none of these, and two of them could never
work there whatever the budget. Measured rather than assumed:

| Back-end | Hosted? | The actual constraint |
|---|---|---|
| **napari** | never | a **desktop Qt application** — it renders into an OS window, not an HTML page. Size is irrelevant |
| **StarDist** | no | needs TensorFlow, which publishes **no wheels for Python 3.14** (3.13 has them). Streamlit Cloud runs 3.14 |
| **Cellpose** | no | works fine on 3.14, but **2.6 GB peak RSS** and a **1.15 GB model download** on first use. The 188 MB wheel is misleading |
| **Gmsh** | no | needs exactly one system library, `libGLU.so.1`. No PyPI package ships it, and a `packages.txt` to apt-install it is what broke the deploy — see [§9](#9-publishing-your-own-copy-on-streamlit) |

So **Docker (or a local install) is where "all of them" lives.** The image
installs Cellpose with a CPU-only PyTorch build; the default wheel bundles
~2.5 GB of CUDA that a CPU image cannot use. Note `torch` and `torchvision` must
come from the **same** index — mixing them installs cleanly and then fails at
import with `RuntimeError: operator torchvision::nms does not exist`.

Cellpose downloads its model on first use, so the first segmentation in a fresh
container needs a network connection and a while.

### Check the install

```bash
pytest                     # 160 tests, ~15 s, must all pass
biosim list                # 4 instruments, 5 solver back-ends
```

`pytest` passing with **no optional back-end installed** is a deliberate design
guarantee, not a coincidence.

### If something is missing

`biosim doctor` never fails — it reports. Each missing component comes with the
capability it would unlock, so you can decide whether you need it.

```
gmsh    missing   python module 'gmsh' not importable (pip install biosim-lab[mesh]);
                  the structured-mesh fallback covers the straight-channel template
```

---

## 4. The web app, page by page

Run it locally:

```bash
streamlit run streamlit_app.py
```

It opens at <http://localhost:8501>.

### Overview

Orientation, plus the three things to know before trusting a number. Start here
if someone else set this up for you.

### SAW cell sorter

The flagship. Every sidebar control re-runs the simulation.

**Controls that change the physics**

| Control | Effect | Watch for |
|---|---|---|
| Channel width | sets which frequency gives one node | the tooltip prints the single-node frequency for the current width |
| Frequency | sets the node spacing (`λ_SAW/2`) | more than one node in the channel and two-outlet sorting stops working |
| Drive voltage | force scales with **voltage squared** | the most effective knob by far |
| Flow rate | sets how long cells spend in the field | raise it and recovery always falls |
| Active length | same effect as flow rate, inversely | this is the IDT aperture in a real chip |
| Force field: analytic / FEM | closed form vs solving the wave equation | FEM is slower and **less optimistic** — see [§11](#11-reading-results-honestly) |
| Temperature | viscosity, and migration speed is inversely proportional to it | 25 → 37 °C makes every cell move ~25 % faster, which *lowers* purity because the background moves too |
| Sample viability | how much of the input was alive to begin with | a fresh suspension is 90–97 % viable, not 100 % |

**Reading the numbers**

- **Recovery** — of the target cells, what fraction reached the collection outlet.
- **Purity** — of what was collected, what fraction is target.
- **Enrichment** — purity divided by the input fraction. The number a rare-cell
  assay is actually judged on. Recovery and purity trade off; enrichment does not
  lie about the trade.
- **Live purity** — of the *live* cells collected, the fraction that are targets.
  A collected dead cell is of no use to whatever comes next, so this is usually
  the honest figure to quote.
- **Viability out** — with the delta showing how many percentage points the
  device itself cost. In this regime it should be zero; if it is not, look at
  the *Cell safety* tab to see which threshold was crossed.

**Tabs**

- *Live view* — **every cell as a moving dot**, seen from above, with a play
  button and a time slider. Marker size follows the real radius; hollow grey
  markers are dead cells. This is the microscope view of the device running.
- *Cross-section* — the channel sliced across at the outlet, cells drawn **to
  scale**. The size difference that drives the whole separation is visible
  directly.
- *Live count* — a running tally at each outlet, as an instrument's counter
  would show it. The slope is throughput in cells per second.
- *Cell safety* — the three damage mechanisms with their published thresholds
  and how far the operating point sits from each.
- *Trajectories* — one line per cell, seen from above.
- *Outlet histogram* — where each cell was when it left. Overlap inside the
  shaded band is what limits purity.
- *Force profile* — the sideways push across the channel, in piconewtons.
- *Size distribution* — the log-normal radii actually drawn.
- *Per-population* — the numbers behind the metrics, plus the contrast factors.
- *Data* — the full per-cell table as CSV, and **the configuration as YAML**,
  which runs unchanged on the command line.

### Impedance (RTCA)

Simulates a 96-well plate: cells attach, grow, get dosed, and an IC50 is fitted
from the endpoint. Or upload a real RTCA export (CSV/XLSX) and it analyses that
instead.

The blue note explaining why the fitted IC50 differs from the planted one is not
an apology — it is the exposure-time effect that real endpoint assays show.

### Cell counter

Segments a synthetic field of view with known ground truth, so the segmentation
can be *scored* rather than admired. Shows the raw image beside the detected
outlines.

The Poisson counting error (`1/√N`) is displayed prominently because it is
usually larger than the difference people are trying to measure.

### Cell tracker

Segments every frame, links detections, and reports speed, directional
persistence and the MSD exponent α (1 = random wandering, 2 = walking
somewhere).

The *Search range* slider is the one that matters: too small and tracks
fragment, too large and identities get swapped. The app warns when you set it
close to the actual step size.

### Material provenance

All 65 physical constants with their source: 27 with a published DOI, 38 flagged
as assumptions with the reason. Downloadable as CSV. This is the list a methods
section should disclose.

### Environment

What is installed, what is not, and what each absent component would have
unlocked. Also reports *how* the instruments were discovered — entry points if
the package is installed, direct import if you are running from a checkout.

---

## 5. The command line

| Command | Does |
|---|---|
| `biosim list` | installed instruments and solver back-ends |
| `biosim doctor` | full environment diagnosis |
| `biosim init <instrument> -o run.yaml` | write a working example config |
| `biosim run run.yaml` | run it, save `.nc` + `.parquet` + `_metrics.csv` |
| `biosim sweep run.yaml -p 'NAME=v1,v2'` | parameter sweep |
| `biosim dashboard run.yaml` | serve the Panel dashboard |
| `biosim materials` | the assumption list |
| `biosim materials --all` | every value with its source |
| `biosim version` | version |

### A complete session

```bash
biosim init saw_sorter -o my_run.yaml
$EDITOR my_run.yaml                     # change frequency, cells, whatever
biosim run my_run.yaml
```

```
ctc_vs_rbc — SAW acoustophoretic cell sorter
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ metric                        ┃     value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ efficiency_percent            │     99.67 │
│ purity_percent                │     95.53 │
│ enrichment_fold               │     1.911 │
│ all_cells_exited              │      True │
│ channel_reynolds              │    0.5335 │
└───────────────────────────────┴───────────┘
RegimeWarning: k*a = 0.34 > 0.1: ...
wrote fields: results/ctc_vs_rbc.nc
wrote table: results/ctc_vs_rbc.parquet
wrote metrics: results/ctc_vs_rbc_metrics.csv
```

### Sweeps

Values may carry units, and repeating `-p` makes the grid multi-dimensional:

```bash
biosim sweep my_run.yaml \
  -p 'voltage_pp=5 V,10 V,15 V,25 V' \
  -p 'flow_rate=5 uL/min,15 uL/min,40 uL/min' \
  -o sweeps/
```

Writes a long table (CSV + Parquet) and an N-dimensional NetCDF ready for a
heatmap.

> **Do not average a sweep across frequency.** Frequency changes the *number* of
> pressure nodes, i.e. the operating regime. `examples/03_parameter_sweep.py`
> produces one heatmap per frequency instead of one averaged over them, for
> exactly this reason.

### Shipped configurations

| File | What it demonstrates |
|---|---|
| `configs/ctc_vs_rbc.yaml` | the reference single-node separation |
| `configs/saw_sorter_20mhz_spec.yaml` | the multi-node regime and its warning |
| `configs/saw_sorter_fem.yaml` | the same device solved by FEM |
| `configs/saw_sorter_whole_blood.yaml` | 1 % tumour cells in a 3-component blood background |
| `configs/rtca_ic50.yaml` | 48 h growth with a drug at 24 h |
| `configs/cell_count_demo.yaml` | synthetic counting |
| `configs/cell_track_demo.yaml` | synthetic tracking |

---

## 6. The Python API

```python
from biosim_lab.core.config import ExperimentConfig
from biosim_lab.registry import installed_instruments

cfg = ExperimentConfig.from_yaml("configs/ctc_vs_rbc.yaml")
Instrument = installed_instruments()[cfg.instrument]
result = Instrument(cfg).run()

result.metrics["efficiency_percent"]   # scalars
result.table.head()                    # one row per cell
result.fields                          # xarray Dataset of trajectories/fields
```

Or drive the simulation directly, without a config file:

```python
from biosim_lab.instruments.saw_sorter.simulate import (
    SAWSorterParams, SAWSorterSimulation,
)

params = SAWSorterParams(
    frequency="6.632 MHz",        # unit strings are validated with pint
    voltage_pp="15 V",
    channel_width="300 um",
    flow_rate="5 uL/min",
    populations=[
        {"cell_type": "mcf7", "count": 300, "target": True},
        {"cell_type": "rbc",  "count": 300, "target": False},
    ],
)
outcome = SAWSorterSimulation(params).run()
print(outcome.metrics["purity_percent"])
```

### Physics functions on their own

```python
from biosim_lab.instruments.saw_sorter.physics.acoustics import (
    contrast_factor, primary_radiation_force_1d, saw_wavelength,
)
from biosim_lab.core.materials import MCF7, WATER

phi = contrast_factor(MCF7.rho, WATER.rho, MCF7.kappa, WATER.kappa)   # 0.2371
lam = saw_wavelength(6.632e6, 3979.0)                                  # 600 µm
```

Every physics function takes **plain SI floats** and returns SI. Units are
checked at the configuration boundary, not inside the numerical core.

### Saving and loading

```python
from biosim_lab.core.io import save_result, load_result

paths = save_result(result, "results/", "my_run")
again  = load_result(paths["fields"])
```

---

## 7. Configuration file reference

Every run is one YAML file:

```yaml
name: my_experiment           # used for output filenames
instrument: saw_sorter        # which plugin
seed: 12345                   # reproducibility; null for random
output_dir: results           # relative paths resolve next to this file
description: >-
  Free text, carried into the result metadata.
params:                       # validated by the instrument's own schema
  frequency: 6.632 MHz
  ...
```

Unit-bearing values are written as strings and checked with `pint`. Writing
`frequency: 300 um` raises a `DimensionalityError` rather than running.

### `saw_sorter`

| Parameter | Default | Meaning |
|---|---|---|
| `frequency` | `6.632 MHz` | IDT drive frequency |
| `voltage_pp` | `15 V` | drive voltage; ignored if `pressure_amplitude` is set |
| `pressure_amplitude` | `null` | measured p₀ — **prefer this if you have it** |
| `substrate` | `linbo3_128yx` | sets the SAW velocity |
| `node_offset` | `null` | node position; defaults to the channel centre |
| `channel_width` | `300 um` | across the acoustic axis |
| `channel_height` | `50 um` | |
| `channel_length` | `2 mm` | the acoustically active length (IDT aperture) |
| `flow_rate` | `5 uL/min` | what a syringe pump controls |
| `fluid` | `water` | `water`, `pbs`, `dmem` |
| `temperature` | `298.15 K` | accepts `"37 degC"`. **Changes migration speed by ~25 % between bench and incubator** |
| `rf_power` | `null` | applied RF power [W], used only for the flagged transducer-heating estimate |
| `inlet_viability` | `0.95` | fraction of the sample already alive before the device |
| `track_viability` | `true` | compute thermal, shear and cavitation damage per cell |
| `populations` | MCF-7 + RBC | list of `{cell_type, count, target}` |
| `tilt_angle_deg` | `0.0` | IDT tilt. 0 = conventional SSAW; non-zero = **tilted-angle SSAW, a different mechanism**. Positive deflects towards −x. Not available with `mode: fem` |
| `inlet` | `sheath_sides` | `sheath_sides`, `uniform`, `centre`, `side` |
| `inlet_side` | `left` | which wall the sample enters on, for `inlet: side` |
| `inlet_band` | `0.15` | inlet stream width, as a fraction of the channel |
| `outlet_layout` | `centre_band` | `centre_band` (three outlets, collect at the node) or `lateral_split` (two outlets, one divider) |
| `collection_fraction` | `0.333` | central outlet width / channel width — `centre_band` only |
| `split_position` | `0.5` | divider position / channel width — `lateral_split` only |
| `collect_side` | `right` | which side of the divider is collected — `lateral_split` only |
| `mode` | `analytic` | `analytic` or `fem` |
| `integration` | `overdamped` | `overdamped` or `inertial` |
| `enable_vertical_arf` | `false` | FEM vertical force — see [§11](#11-reading-results-honestly) |
| `enable_gravity` | `false` | sedimentation |
| `enable_wall_repulsion` | `false` | numerical regulariser, not physics |
| `enable_secondary_bjerknes` | `false` | cell–cell acoustic interaction, `O(N²)` |
| `fem_resolution` | `40` | elements across the channel width |
| `fem_grid` | `[241, 41]` | sampling grid for the force field |
| `n_time_samples` | `101` | trajectory sample points |
| `seed` | `12345` | |

Cell types: `mcf7`, `hela`, `a549`, `rbc`, `wbc`, `platelet`, `ps_bead`, `lipid`.

### `impedance_rtca`

| Parameter | Default | Meaning |
|---|---|---|
| `frequency` | `10 kHz` | Cell Index readout frequency |
| `spectrum_frequencies` | `60` | points in the `|Z|(f)` sweep |
| `spectrum_range` | `[100, 1e7]` | Hz |
| `n_wells` | `96` | `96` or `384` |
| `duration` | `48 h` | |
| `n_timepoints` | `97` | |
| `doubling_time` | `20 h` | |
| `lag_time` | `2 h` | attachment delay before growth starts |
| `seeding_coverage` / `max_coverage` | `0.05` / `0.95` | electrode coverage limits |
| `treatment_time` | `24 h` | when the drug goes in; `null` for none |
| `concentrations` | 8-point series | dose series |
| `replicates` | `3` | wells per dose |
| `true_ic50` / `hill_slope` | `1.0` / `1.3` | used to synthesise the plate |
| `noise_cv` | `0.02` | measurement noise |
| `conductivity` | `1.4 S/m` | medium |
| `junction_resistance` | `2.0` | `R_b`, Ω·cm² |
| `membrane_capacitance` | `1e-6` | F/cm² |
| `electrode_area_cm2` | `0.008` | |
| `cell_radius` / `gap_height` | `8 um` / `100 nm` | shell-model geometry |
| `source_file` | `null` | path to a real RTCA export |

### `cell_counter`

| Parameter | Default | Meaning |
|---|---|---|
| `source_image` | `null` | TIFF/PNG; synthesises when absent |
| `pixel_size` | `0.65 um` | metres per pixel |
| `chamber_depth` | `100 um` | Neubauer standard |
| `dilution_factor` | `2.0` | 1:1 trypan-blue mix |
| `backend` | `classical` | `classical`, `cellpose`, `stardist` |
| `min_radius_px` | `4.0` | debris cut-off |
| `min_diameter_um` / `max_diameter_um` | `5` / `40` | size gate |
| `viability_threshold` | `null` | `null` uses Otsu on intensity |
| `n_cells`, `dead_fraction`, `image_size`, `seed` | | synthetic-demo controls |

### `cell_tracker`

| Parameter | Default | Meaning |
|---|---|---|
| `source_movie` | `null` | TIFF stack or image directory |
| `pixel_size` | `0.65 um` | |
| `frame_interval` | `10 min` | |
| `backend` | `classical` | |
| `search_range_px` | `12.0` | **the parameter that matters most** |
| `memory_frames` | `2` | frames a cell may vanish for |
| `min_track_length` | `5` | shorter tracks are discarded |
| `n_frames`, `n_cells`, `image_size`, `speed_px_per_frame`, `persistence`, `seed` | | synthetic-demo controls |

---

## 8. Using your own data

### Real RTCA exports

```yaml
instrument: impedance_rtca
params:
  source_file: /path/to/RTCA_export.csv
```

The reader skips metadata rows above the header, tolerates rows with different
field counts, sniffs the delimiter, and locates the header by finding the first
row containing well labels (`A1` … `H12`). CSV and XLSX both work.

Or drop the file into the web app's uploader.

### Microscopy images

```yaml
instrument: cell_counter
params:
  source_image: /path/to/field.tif
  pixel_size: 0.325 um        # YOUR objective and camera, not the default
  chamber_depth: 100 um
  dilution_factor: 2.0
```

For tracking, `source_movie` accepts a multi-page TIFF **or a directory** of
numbered images.

> **Set `pixel_size` correctly or every physical number is wrong.** It converts
> pixels to metres, so it propagates into diameter, concentration and speed. Get
> it from a stage micrometer, not from the objective's nominal magnification.

### Temperature

Everything in the platform used to assume 25 °C silently. It no longer does, and
the difference is large enough to matter:

```yaml
params:
  temperature: 37 degC        # or "310.15 K", or "98.6 degF"
```

| Temperature | Viscosity | Migration speed | Purity in the reference run |
|---|---|---|---|
| 4 °C | 1.568 mPa·s | 0.55× | 97.6 % |
| 25 °C | 0.890 mPa·s | 1.00× | 95.2 % |
| 37 °C | 0.691 mPa·s | 1.25× | 93.0 % |

Note the direction: warming makes cells migrate faster, which **lowers** purity,
because the background population migrates faster too and more of it reaches the
collection outlet. Cooling is a purity knob. That is not obvious from the
equations and is exactly the kind of thing the simulation is for.

If you know the RF power going into the transducer, `rf_power` adds a flagged
estimate of the resulting temperature rise. The heat the *water* absorbs is
computed from first principles and is negligible (0.006 K at the reference
point); the transducer term is an assumption and is the one that dominates in a
real chip. Measure your device and set `temperature` directly rather than
relying on it.

### Measured acoustic pressure

If you have calibrated your chip, skip the voltage assumption entirely:

```yaml
params:
  voltage_pp: null
  pressure_amplitude: 0.38 MPa    # measured, e.g. by bead tracking
```

---

## 9. Publishing your own copy on Streamlit

The repository already contains everything Streamlit Community Cloud needs.

| File | Role |
|---|---|
| `streamlit_app.py` | the entry point and router, at the repository root |
| `biosim_lab/app/` | the app itself: one module per page under `app/pages/`, cached simulation wrappers in `app/runners.py`, formatting helpers in `app/shared.py` |
| `requirements.txt` | the Python dependencies — **the full set that works on a managed host** |
| `.streamlit/config.toml` | theme matching the figure palette, upload limit |

There is deliberately **no `packages.txt`**, and adding one is the single
easiest way to take this deployment down.

### Why there is no `packages.txt`

Its mere presence makes Streamlit Cloud run `apt-get update` before pip. The
base image carries repositories this project does not control, and one of them
expiring is enough to sink the deploy:

```
E: Release file for .../bullseye-security/InRelease is expired (invalid since 12h)
❗️ installer returned a non-zero exit code
❗️ Error during processing dependencies!
```

The consequence is worse than losing a feature. The new instance never starts,
so the **previous process keeps serving** — executing the old bytecode against
the newly pulled source. A fix you just pushed appears to change nothing, and
the traceback mixes old line numbers with new source text. The app now detects
that state and says so on every page (see §12).

Nothing here needs a system library. Gmsh is the one dependency that cannot
import without OpenGL, and it is optional by design: meshing falls back to the
structured straight-channel template, and `biosim doctor` reports it missing.
Everything else in `requirements.txt` imports unaided — the `streamlit` CI job
installs it with no apt step at all, precisely so this fails in CI rather than
in a hosting build log.

If you do add a system dependency, note that **the parser has no comment
support**: it splits on whitespace and hands every token to `apt-get install`,
so one explanatory comment becomes `E: Unable to locate package OpenGL,`. One
bare package name per line, nothing else.

| Main file path | `streamlit_app.py` |
| Python version | 3.11 or 3.12 |

Press *Deploy*. The first build takes 3–5 minutes while it installs the
dependencies; later pushes to `main` redeploy automatically.

**3. Check it.** Open the *Environment* page in the deployed app. It should say:

```
Instruments found via direct import (4 built-in instrument(s));
the package is not pip-installed, so third-party plugins will not be discovered
```

That is **expected and correct**. Streamlit Cloud installs `requirements.txt`
but does not `pip install` the project itself, so entry-point metadata is
absent; `biosim_lab.registry` falls back to importing the built-in instruments
directly. Everything works; only third-party plugins would be missed.

### What is installed, and what cannot be

`requirements.txt` carries everything that genuinely functions on a managed
host, verified by installing it in a `linux/amd64` Debian bookworm container —
the same platform Streamlit Cloud runs — and then meshing, rendering, writing
NetCDF and running the full test suite inside it.

Three things are absent on purpose, and adding them to `requirements.txt` would
not help:

| Absent | Why adding it would not help |
|---|---|
| **Napari** | needs Qt and a display server. It would install, report itself present, and still be unable to open a viewer. Use it locally: `pip install -e ".[imaging]"`, then `instrument.view_napari()`. |
| **OpenFOAM, Elmer** | external binaries, not Python packages, from apt repositories a managed host does not carry. Their absence means acoustic streaming is not modelled and the analytic Rayleigh approximation is used, and the piezoelectric problem is replaced by the documented voltage calibration. Use `docker/Dockerfile.openfoam` and `docker/Dockerfile.elmer`. |
| **Cellpose, StarDist** | installable, but they pull a deep-learning runtime. The default PyTorch wheel bundles CUDA at roughly 2.5 GB, which will not fit a free tier. |

To enable Cellpose anyway on CPU, uncomment the four lines at the bottom of
`requirements.txt`:

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch
torchvision
cellpose>=3.0
```

That is about 250 MB and inference is slow without a GPU. The classical
watershed back-end needs none of it and is what every figure in the app uses.

The app's **Environment** page reports all of this live, split into *installed*,
*missing but fixable* and *cannot work here* — so you never have to guess which
kind of absence you are looking at.

### Staying inside the free tier

The free tier gives roughly 1 GB of RAM.

- **Every simulation is cached** on its parameters (`st.cache_data`), so
  returning to a previous slider position is instant and the host is not asked
  to recompute the same thing twice.
- **The sliders are capped**: 600 cells per population, mesh resolution 64,
  30 tracker frames, 512-pixel images. Raise them in `biosim_lab/app/pages/` if you
  are hosting somewhere larger.
- **PyVista is the heaviest import.** It is present and works off-screen, but
  importing VTK costs a few hundred megabytes of resident memory. If the app is
  being killed, dropping `pyvista` from `requirements.txt` is the single biggest
  saving; nothing in the current pages depends on it.

If the app is killed for memory, the other usual cause is FEM mode at high mesh
resolution combined with a large cell count. Lower `fem_resolution` first.

### One trap worth knowing about

Streamlit runs your script on a **worker thread**, not the main thread. Gmsh
installs a SIGINT handler on `initialize()`, and Python only permits that from
the main thread, so a naive `gmsh.initialize()` inside a Streamlit app fails
with `signal only works in main thread of the main interpreter` — and Gmsh looks
permanently broken even though it is installed correctly.

`biosim_lab.core.geometry` handles this: it detects the thread and passes
`interruptible=False` when it is not on the main one. The only thing lost is
Ctrl-C during meshing, which is meaningless in a web app. The same fix makes
Gmsh work inside Jupyter kernels, Dask workers and web request handlers.

### Running the web app locally instead

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Anything the hosted app can do, this can do — plus, if you have the full install,
the extras the Environment page lists as missing.

---

## 10. Docker

```bash
docker compose up biosim              # Panel dashboard on http://localhost:5006
docker compose run --rm tests         # the test suite
docker compose run --rm examples      # every example, writes assets/
docker compose run --rm cli           # biosim doctor
```

The heavy solvers sit behind a profile so a plain `up` never builds them:

```bash
docker compose --profile solvers build openfoam elmer
```

The main image deliberately contains neither OpenFOAM nor Elmer. On `linux/arm64`
it also builds without Gmsh, since no wheel exists there — the structured-mesh
fallback covers everything the tests and examples use, and all 160 tests pass in
the container.

---

## 11. Reading results honestly

This section is the one that stops you publishing something wrong.

### The warnings are the product

Orange `RegimeWarning` banners mean an assumption behind the model has been
stretched. They are not noise:

| Warning | Means | Do |
|---|---|---|
| `k*a > 0.1` | cells are not small compared with the sound wavelength; the force is over-predicted | treat forces as indicative, not absolute |
| `N pressure nodes` | the channel holds more than one collection line | change frequency or width, or accept a multi-outlet design |
| `Reynolds number > 1` | the creeping-flow model no longer applies | lower the flow rate, or use the OpenFOAM back-end |
| `Stokes number > 0.1` | particle inertia matters | switch `integration` to `inertial` |

### `analytic` is the optimistic mode

The closed-form model applies the sound strength at the chip surface to cells at
*every* height. The real field weakens upwards — averaged across a 50 µm channel
the lateral force is only **53 %** of the surface value.

| | recovery | purity |
|---|---|---|
| `mode: analytic` | 100 % | 94.3 % |
| `mode: fem` | 45.5 % | 98.9 % |

Neither is a bug. Use `analytic` to explore a design space quickly, then confirm
with `fem` before believing a number. `examples/04_validate_analytic_vs_fem.py`
prints the comparison and explains every source of the gap.

### `enable_vertical_arf` will strand your cells

It is `false` by default. The FEM field has a genuine vertical force, but this
2-D cross-section model has no lift force to balance it, so cells pile against a
wall where the axial flow is zero and never reach the outlet.

If you turn it on, **check `all_cells_exited` in the metrics.** If it is `False`,
the recovery and purity numbers are meaningless. The web app shows a red banner
in this case.

### Viability is not free of assumptions either

The *Cell safety* tab computes three published damage indicators and how far the
operating point sits from each. At the reference point all three margins are
comfortable, and the dominant reason is exposure time — cells are in the field
for 0.36 s. **A trap that held them for minutes at the same intensity would be a
completely different proposition**, and the same code will tell you so.

What is *not* modelled: membrane poration below the lysis threshold, which can
let trypan blue into a cell without killing it and therefore corrupts a
viability readout; and any change in a cell's acoustic properties once it dies.
Dead cells are propagated with live-cell density and compressibility, so their
predicted destination is less trustworthy than that of live ones.

`inlet_viability` defaults to 0.95, not 1.0, because a real suspension is never
fully viable. If you compare against an experiment, set it to what your
haemocytometer actually said.

### Check the provenance before quoting a number

```bash
biosim materials            # the 38 assumptions, with reasons
```

The largest single one: nothing here predicts acoustic pressure from drive
voltage. A linear calibration (15 Vpp → 0.45 MPa) stands in, and it scales
*every* acoustic force. Measure it for your chip and set `pressure_amplitude`.

### Not modelled at all

Acoustic streaming (matters below ~1 µm particles), cell–cell acoustic
interaction, cell deformability, and the membrane/nucleus structure for
acoustics. `docs/physics.md` §5 has the full list with references.

### Two-outlet sorting: large cells one side, everything else the other

The default chip has **three** outlets and creams the large cells off the middle,
at the pressure node. Many real devices are **two**-outlet instead: the sample
enters along one wall, large cells cross the channel toward the node, small ones
do not, and one divider separates the two streams. Set:

```yaml
params:
  inlet: side            # the whole sample hugs one wall
  inlet_side: left
  outlet_layout: lateral_split
  split_position: 0.45   # divider, as a fraction of the channel width
  collect_side: right    # which side is the collection outlet
```

or pick *Outlet layout → lateral_split* in the web app, which reveals a divider
slider in place of the collection-band width.

**The divider does not go on the node.** Cells approach a node asymptotically and
settle a few microns short of it, so a divider placed exactly on the node
collects nothing — the model warns rather than reporting a bare 0 % recovery.
Put it between the two populations' landing positions, which
`examples/09_two_outlet_split.py` sweeps for you:

| divider (µm) | recovery | purity |
|---|---|---|
| 60 | 100.0 % | 73.9 % |
| 90 | 100.0 % | 92.3 % |
| 120 | 98.0 % | 98.7 % |
| 135 | 92.3 % | 99.6 % |
| 150 (on the node) | 0.0 % | — |

That is the recovery-versus-purity trade-off made explicit: the divider is the
knob, and there is no setting that maximises both. In this geometry MCF-7 land
at 144 ± 8 µm and red cells at 53 ± 24 µm, so anywhere from 90 to 135 µm is
defensible depending on which you care about.

A one-side inlet is worth having on its own, incidentally: it gives every cell
the full channel width to migrate across, so the separation is as strong as the
geometry allows, whereas `sheath_sides` starts cells at *both* walls and halves
the distance available.

### Tilted-angle SSAW: a different mechanism, not a different outlet

Everything above assumes the standing wave runs **straight** across the channel.
The node planes then lie parallel to the flow, a cell migrates sideways until it
reaches a node, and it **stops**. Its displacement is capped by the node spacing
no matter how long the channel is or how strong the field.

Tilting the IDTs (`tilt_angle_deg`, doi:10.1073/pnas.1413325111) changes what the
device does. The node planes now cross the flow, so a cell held in one is dragged
across the channel as it travels downstream:

```
dx/dz = -tan(theta)      so a held cell drifts  L * tan(theta)  over length L
```

Displacement grows with **channel length** instead of saturating, and the
separation stops being "how fast does it migrate" and becomes **"can a node hold
it at all"**.

**The design number is the tilt limit.** Holding a cell on a moving node plane
costs a sideways drag of `u * tan(theta)`, so a cell stays trapped only while

```
sin(theta) / cos^2(theta)  <=  pi * p0^2 * kappa_f * Phi * a^2 / (9 * mu * lambda * u)
```

That right-hand side scales with **a²**, so small cells lose their grip first —
and *that asymmetry is the separation*. `biosim_lab.instruments.saw_sorter.
physics.acoustics.max_trappable_tilt` gives each population's limit and
`cutoff_radius` inverts it into the sorter's cutoff size; both appear in
`diagnostics["tilt"]` and on the web app when the angle is non-zero. At
6.632 MHz, 15 Vpp and 5 µL/min:

| cell | radius | holds up to |
|---|---|---|
| MCF-7 | 9.00 µm | 16.5° |
| A549 | 7.75 µm | 9.8° |
| WBC | 4.25 µm | 2.6° |
| RBC | 2.78 µm | 2.4° |

Any angle between 2.4° and 16.5° deflects the tumour cells and lets the blood
cells flow straight through. Pick one and the cutoff diameter follows: 10° gives
13.7 µm.

**Past the limit the device silently does nothing.** A cell that cannot be held
slips across node planes, the force averages to zero, and it flows on almost
undeflected — a failure that produces perfectly plausible-looking output. Check
the limit rather than assuming a bigger angle deflects harder.

**Two things the model refuses or flags.** `mode="fem"` with a tilt raises: the
Helmholtz field is solved on the channel cross-section and does not vary along
the flow, whereas a tilted pattern varies along the flow by definition, so that
mesh would silently return a straight-IDT answer under a tilted label. And a
positive angle deflects towards −x, so with the sample on the left wall you want
a **negative** angle; get the sign wrong and every cell is pressed into the wall
it started against, which the model warns about.

`examples/10_tilted_angle_ssaw.py` shows both the loss of saturation and what it
buys: in a 600 µm channel one wavelength wide, a straight device cannot reach
90 % recovery at any divider, while −10° gives 100 % recovery at 100 % purity.

### Which of the 38 assumptions actually matter

The Material provenance page lists every number that could not be traced to a
DOI. That is an honest disclosure but not an actionable one: it says what is
unknown, not what the ignorance costs. To rank them:

```bash
python examples/07_assumption_sensitivity.py
```

It perturbs each unsourced value by ±5 % and reports the **normalised
elasticity** `(dY/Y)/(dX/X)` — dimensionless, so a density and a viscosity can
be compared on one axis. An elasticity of 1 means a 10 % error in that input
gives a 10 % error in the answer; 0 means it does not matter.

Two results are worth knowing before you run it:

* **Exactly one assumption moves the CTC/RBC separation**: the MCF-7 cell
  density (elasticity 0.94). Cell-radius spread is a distant second at −0.16.
  Everything else is either unused by the acoustic model or below the resolution
  of the measurement. So the answer to "what should I measure first?" is one
  item, not thirty-eight — band your cell line on a density gradient.
* **The ranking is a property of the operating point, not of the model.** At the
  design point the sorter recovers essentially every target cell, and sitting
  against that ceiling makes it insensitive to everything (largest elasticity
  0.007). At a marginal point the same number matters **130× more**. Quoting a
  sensitivity without the operating point it was measured at is meaningless —
  run the scan where you actually operate.

The scan is careful about two things that are easy to get wrong. Perturbed runs
reuse the baseline's random seed, so the difference isolates the parameter
instead of measuring resampling noise; and significance is judged on the *paired
difference* across seeds, not on the spread of the raw metric, which is the
wrong yardstick by a large factor and rejects real effects.

### Chaining instruments, and the error budget

Sorting, counting and tracking are separate instruments here, but a real
experiment runs them in sequence. Two things only become visible when they are
chained:

```bash
python examples/08_pipeline_sort_count_track.py
```

**The sorter changes the population, not just its size.** Radiation force scales
with cell volume while drag scales with radius, so migration speed goes as `r²`
and collection is size-selective. In the worked example the loaded suspension has
a mean diameter of 11.7 µm with a CV of 0.54; what reaches the counter is
18.3 µm with a CV of 0.12 — **+56 % in the mean and 4.5× narrower**. A counter
gated on the loaded distribution would be measuring the wrong population.
`CountStage` therefore takes its size distribution from the sample it is handed,
not from a default.

**The uncertainties compose, and one stage dominates.** Each stage contributes a
different kind of error: sorting a *binomial* one (a finite number of cells
either reach the outlet or do not), counting a *Poisson* one (`1/√N` cells in the
field of view), tracking a *sample-to-sample* one. They are independent, so they
add in quadrature — 3 % and 4 % make 5 %, not 7 % — and the total is usually
dominated by one stage. The summary names it. That is the actionable part:
imaging more fields of view cannot rescue a sorting-limited measurement.

---

## 12. Troubleshooting

**`biosim: command not found`**
The virtual environment is not active. `source .venv/bin/activate`, or call it
as `python -m biosim_lab.cli`.

**`0 instruments discovered`**
The package was not installed. Either `pip install -e .`, or use
`biosim_lab.registry.installed_instruments()`, which falls back to importing the
built-ins.

**`RuntimeError: a PDMS wall layer requires gmsh`**
`pip install biosim-lab[mesh]`. Without Gmsh only the wall-free straight channel
is available — which is what every shipped configuration uses.

**`no usable NetCDF back-end`**
`pip install netCDF4` (or `h5netcdf` *and* `h5py` — `h5netcdf` imports without
`h5py` but fails at write time). Or write CSV instead.

**The simulation is very slow**
Almost always `mode: fem` with a high `fem_resolution`. Start at 32. If the
*analytic* mode is slow, check `all_cells_exited`: cells stuck against a wall
extend the integration window enormously.

**Results changed after I set a temperature**
They should have. Viscosity falls 22 % between 25 °C and 37 °C and migration
speed is inversely proportional to it. Before, everything was implicitly at
25 °C. If you want the old numbers, set `temperature: 25 degC` explicitly.

**`DimensionalityError`**
A unit in the YAML has the wrong dimension — e.g. `frequency: 300 um`. This is
the unit checker doing its job.

**Every cell ends up in one place, no separation**
Look at the diagnostics line. If there are several pressure nodes the channel
sorts into stripes, not two outlets. Use `f = c_SAW / (2 × width)`.

**The Streamlit app is killed on the free tier**
Lower `fem_resolution` and the cell count. See [§9](#9-publishing-your-own-copy-on-streamlit).

**Deployment fails with `E: Unable to locate package <an English word>`**
There is a comment in `packages.txt`. The parser has no comment support and is
reading your prose as package names. Strip it to bare package names, one per
line — `pytest tests/test_deployment.py` checks this.

**Deployment fails with `libGL.so.1: cannot open shared object file`**
The opposite problem: `packages.txt` is missing or does not list `libgl1`.
PyVista and Gmsh link against OpenGL even when rendering off-screen.

**Tests fail after I changed something**
Read *which* test. They check relationships (force ∝ r³, Φ > 0 for cells, no
cell lost) rather than remembered numbers, so a failure usually names the
physical property you broke.

---

## 13. Extending it

Adding an instrument takes ten steps and **no change to the core** — the full
walkthrough is in [`CONTRIBUTING.md`](../CONTRIBUTING.md). In outline:

1. Make a package under `biosim_lab/instruments/` (or in your own distribution).
2. Write a `BaseConfigModel` subclass with unit-annotated fields.
3. Put the physics in its own module, with a DOI on every formula.
4. Warn with `RegimeWarning` when the model is out of range.
5. Implement `Instrument`: `setup()` (idempotent) and `run()`.
6. Use the core `io` and `viz` layers rather than your own.
7. Add a dashboard using `core.viz.dashboard.shell()`.
8. Provide a working `example_config()` — the tests execute it.
9. Register an entry point under `biosim_lab.instruments`.
10. Write tests, including one that skips cleanly if an optional back-end is absent.

To add a **solver** back-end instead, implement `Solver`, declare
`required_executables` / `required_modules`, and register under
`biosim_lab.solvers`. When it is absent, `get_solver()` returns an
`UnavailableSolver` that explains itself — never an `ImportError`.

### The rules that must not be broken

1. The whole test suite passes with no optional back-end installed.
2. Every physical constant has a DOI or an explicit `ASSUMPTION` label.
3. Units are validated at the boundary; the core uses plain SI floats.
4. Instruments never import each other, and the core imports no instrument.
5. Out-of-range models warn; they never quietly return a plausible wrong number.

---

## Further reading

| Document | Contents |
|---|---|
| [`docs/explainer/biosim-lab-explained.pdf`](explainer/) | 29-page explainer for non-experts |
| [`ARCHITECTURE.md`](../ARCHITECTURE.md) | layers, contracts, data flow, limitations |
| [`docs/physics.md`](physics.md) | every formula, its source, and where it stops being valid |
| [`docs/validation.md`](validation.md) | what was checked against what, and to what tolerance |
| [`docs/instruments.md`](instruments.md) | per-instrument reference (Turkish) |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | adding an instrument in ten steps |
