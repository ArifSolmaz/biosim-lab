"""Example 6 — cell counting and tracking on synthetic microscopy (Stage 3).

Generates a field of view and a time-lapse with known ground truth, runs the
watershed segmentation and the trackpy linking, and scores the result against
that ground truth — which is the point of the synthetic generator: without it
there is nothing to score against.

Run::

    python examples/06_imaging_demo.py
"""

from __future__ import annotations

from pathlib import Path

from biosim_lab.core.config import ExperimentConfig
from biosim_lab.core.io import save_result
from biosim_lab.core.viz.napari_layers import napari_available
from biosim_lab.instruments.cell_counter import CellCounter
from biosim_lab.instruments.cell_counter.segmentation import available_backends
from biosim_lab.instruments.cell_tracker import CellTracker

OUT_DIR = Path(__file__).resolve().parent.parent / "assets"


def counter_demo() -> None:
    print("=" * 72)
    print("Automated cell counter — synthetic brightfield field of view")
    print("=" * 72)
    print("  segmentation back-ends:")
    for name, (ok, detail) in available_backends().items():
        print(f"    {name:12s} {'available' if ok else 'missing':10s} {detail}")
    print()

    cfg = ExperimentConfig.model_validate(CellCounter.example_config())
    instrument = CellCounter(cfg)
    result = instrument.run()
    m = result.metrics

    print(f"  ground truth cells    {m['ground_truth_n']:8d}")
    print(f"  detected              {m['n_total']:8d}   (recall "
          f"{m['detection_recall'] * 100:.1f} %)")
    print(f"  live / dead           {m['n_live']:5d} / {m['n_dead']:d}")
    print(f"  viability             {m['viability_percent']:8.1f} %   "
          f"(ground truth {m['ground_truth_viability_percent']:.1f} %)")
    print(f"  concentration         {m['concentration_per_ml']:8.3g} cells/mL")
    print(f"  sampled volume        {m['sampled_volume_ml']:8.3g} mL")
    print(f"  mean diameter         {m['mean_diameter_um']:8.2f} um "
          f"(CV {m['cv_diameter_percent']:.1f} %)")
    print(f"  Poisson counting error {m['counting_relative_uncertainty'] * 100:7.1f} %")
    print()
    print("  Detection recall below 100 % is expected: cells touching the frame edge")
    print("  are discarded (they have no valid area) and a few touching pairs merge.")
    print("  Both effects are what a real counter does, and the size gate is what")
    print("  keeps a merged doublet from being read as one giant cell.")

    written = save_result(result, OUT_DIR, "06_cell_count")
    for kind, path in written.items():
        print(f"  wrote {kind}: {path}")


def tracker_demo() -> None:
    print()
    print("=" * 72)
    print("Live-cell tracker — synthetic time-lapse")
    print("=" * 72)

    cfg = ExperimentConfig.model_validate(CellTracker.example_config())
    instrument = CellTracker(cfg)
    result = instrument.run()
    m = result.metrics

    print(f"  frames                {m['n_frames']:8d}")
    print(f"  detections            {m['n_detections']:8d}")
    print(f"  tracks                {m['n_tracks']:8d}   "
          f"(ground truth {m['ground_truth_n_tracks']}, recovery "
          f"{m['track_recovery_ratio']:.2f} x)")
    print(f"  mean track length     {m['mean_track_length_frames']:8.1f} frames")
    print(f"  mean speed            {m['mean_speed_um_per_min']:8.4f} um/min "
          f"(planted {m['ground_truth_speed_um_per_min']:.4f})")
    print(f"  median persistence    {m['median_persistence']:8.3f}")
    print(f"  MSD exponent alpha    {m['msd_alpha']:8.2f}   "
          f"(1 = diffusive, 2 = ballistic)")
    print(f"  diffusion coefficient {m['diffusion_coefficient_m2_s']:8.3g} m^2/s")
    print()
    print("  alpha well above 1 is the signature of persistent migration: the motion")
    print("  is ballistic at short lag and only randomises over many frames.")

    ok, detail = napari_available()
    print(f"\n  napari: {'available' if ok else 'not installed'} — {detail}")
    if not ok:
        print("  (instrument.view_napari() would open the movie with a tracks layer;")
        print("   install with `pip install biosim-lab[imaging]`)")

    written = save_result(result, OUT_DIR, "06_cell_track")
    for kind, path in written.items():
        print(f"  wrote {kind}: {path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counter_demo()
    tracker_demo()


if __name__ == "__main__":
    main()
