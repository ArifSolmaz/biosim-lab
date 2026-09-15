"""Count a sort from its video: the step both reference papers actually took.

Capture efficiency in Zhang et al. (2023, doi:10.3390/ijms24043338) is not a
number the chip reports. It was counted in a microscope video: cells that
"converged to the midline of the microchannel", told apart "by size". Li et
al. (2015, doi:10.1073/pnas.1504484112) counted calcein-stained cells by
fluorescence. So a simulated capture efficiency and a measured one differ by
more than physics --- the counting has errors of its own.

This script puts the counting step back (``biosim_lab.video_readout``):

1. Simulate the Zhang device (``benchmarks/benchmark_02_alternating_baw``).
2. Film the last 250 um before the outlets: brightfield plus a fluorescence
   channel in which only the cancer cells are stained, cells arriving as a
   continuous stream locked to the relay's switching clock.
3. Segment every frame (watershed), link the detections into tracks
   (trackpy), classify each track by size or by stain, and read its outlet
   from where it leaves the field of view.
4. Compute capture efficiency and contamination from those counts alone,
   and only then compare them with the simulation, cell by cell.

Then the same sort is filmed at increasing cell densities, which shows how
dilute the stream has to be before a video count can be trusted.

Writes ``assets/13_*``. Takes a few minutes (tens of thousands of frames).

Run::

    python examples/13_sorter_video_readout.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # for the benchmark's config helpers

from benchmarks.benchmark_02_alternating_baw import cases  # noqa: E402
from benchmarks.common import parallel_map  # noqa: E402
from biosim_lab.core.plugin import RegimeWarning  # noqa: E402
from biosim_lab.core.viz.curves import (  # noqa: E402
    readout_comparison_figure,
    video_frame_figure,
)
from biosim_lab.video_readout import CameraSpec, count_film, film_sorter  # noqa: E402

ASSETS = ROOT / "assets"
SEED = 13
KEYS = [("capture_efficiency_percent", "capture efficiency"),
        ("contamination_rate_percent", "contamination")]


def crowding_case(args: tuple) -> dict:
    """Film a fixed subset of the same sort at one density and score the count."""
    params, outcome, in_view = args
    film = film_sorter(params, outcome=outcome, seed=SEED, n_cells=120,
                       camera=CameraSpec(cells_in_view=in_view), keep_frames=0)
    r = count_film(film, "size")
    e, v, t = r.error_budget, r.video_metrics, r.truth_metrics
    return {
        "cells_in_view": in_view,
        "frames": film.geometry["n_frames"],
        "counted_percent": e["counted_fraction_percent"],
        "missed_occluded": e["missed_occluded"],
        "missed_other": e["missed_other"],
        "extra_tracks": e["extra_tracks"],
        "missed_contaminants_percent": e["missed_background_collected_percent"],
        "missed_waste_background_percent": e["missed_background_waste_percent"],
        "identity_swaps": e["identity_swaps"],
        "classified_correctly_percent": e["classification_accuracy_percent"],
        "capture_video": v["capture_efficiency_percent"],
        "capture_simulation": t["capture_efficiency_percent"],
        "contamination_video": v["contamination_rate_percent"],
        "contamination_simulation": t["contamination_rate_percent"],
    }


def main() -> None:
    warnings.simplefilter("ignore", RegimeWarning)
    ASSETS.mkdir(exist_ok=True)

    # 1-3. The paper's device at its Fig. 2b,c baseline, filmed once.
    params = cases.make_params()
    film = film_sorter(params, seed=SEED, keep_frames=60, keep_stride=6)
    g = film.geometry
    print(f"filmed {g['n_cells_filmed']} cells: {g['n_frames']} frames at "
          f"{g['frame_rate_hz']:.0f} fps, {g['rows']} x {g['cols']} px of "
          f"{g['pixel_size_m'] * 1e6:.1f} um; search radius {g['search_range_px']:.1f} px")

    # 4. Count it twice from the same video: by size (Zhang) and by stain (Li).
    rows = []
    readouts = {}
    for by in ("size", "fluorescence"):
        r = count_film(film, by)
        readouts[by] = r
        v = r.video_metrics
        print(f"\ncounted by {by} (threshold {v['threshold']:.3g} {v['threshold_units']}):")
        print("  " + r.summary().replace("\n", "\n  "))
        rows.append({"classified_by": by, **{f"video_{k}": v[k] for k, _ in KEYS},
                     **{f"simulation_{k}": r.truth_metrics[k] for k, _ in KEYS},
                     **r.error_budget})
    e = readouts["size"].error_budget
    print(f"\n  where the count loses cells: {e['missed_cells']} of "
          f"{e['cells_through_window']} never counted, {e['missed_occluded']} of them "
          f"occluded ---\n  merged with or tangled in another cell (the acoustic nodes "
          f"line cells up, and\n  cells at different heights overtake one another); "
          f"{e['extra_tracks']} extra tracks; {e['identity_swaps']} identity swaps.")
    print(f"  PBMCs that went to the collection outlet were missed "
          f"{e['missed_background_collected_percent']:.0f} % of the time, those that "
          f"went to waste\n  {e['missed_background_waste_percent']:.0f} %: the "
          f"contaminants share the midline with the large MCF-7s and\n  hide behind "
          f"them, so a video count under-reports contamination.")
    pd.DataFrame(rows).to_csv(ASSETS / "13_video_readout.csv", index=False)

    # Figures: the comparison, one frame with its tracks, and the video itself.
    r = readouts["size"]
    readout_comparison_figure(r.video_metrics, r.truth_metrics, KEYS,
                              title="Zhang 2023 device: counted from the video vs simulated"
                              ).write_html(ASSETS / "13_video_vs_simulation.html",
                                           include_plotlyjs="cdn")
    last = g["sample_first_frame"] + (len(film.sample_frames) - 1) * g["keep_stride"]
    links = film.links[film.links["frame"].between(last - 400, last)].copy()
    called = r.tracks["is_target"].to_dict()
    links["label"] = links["particle"].map(
        lambda p: "not counted" if p not in called else ("MCF7" if called[p] else "PBMC"))
    frame = video_frame_figure(
        film.sample_frames[-1], pixel_size=g["pixel_size_m"], tracks=links,
        collection_band=g["collection_um"], margin_px=g["margin_px"],
        exit_col=0.85 * g["cols"], colors={"not counted": "#9a9893"},
        title="Outlet region, brightfield, with the tracks so far (classified by size)")
    frame.write_html(ASSETS / "13_video_frame.html", include_plotlyjs="cdn")
    try:
        frame.write_image(ASSETS / "13_video_frame.png", scale=2)
    except Exception as exc:  # kaleido is optional
        print(f"  (no PNG: {exc})")
    try:
        import imageio.v2 as imageio

        # Crop to the rows cells actually use and average 2x2 blocks: sensor
        # noise is what makes a GIF large, and the cells stay easy to see.
        both = np.concatenate([film.sample_frames, film.sample_fluorescence], axis=2)[:45]
        used = film.truth["y"]
        top = max(0, int(used.min()) - 30)
        bottom = min(g["rows"], int(used.max()) + 30)
        crop = both[:, top:bottom - (bottom - top) % 2, :both.shape[2] // 2 * 2]
        small = crop.reshape(crop.shape[0], crop.shape[1] // 2, 2, crop.shape[2] // 2, 2)
        small = small.mean(axis=(2, 4))
        imageio.mimsave(ASSETS / "13_video.gif", (np.clip(small, 0, 1) * 255).astype(np.uint8),
                        duration=0.08, loop=0)
        print(f"  wrote {ASSETS / '13_video.gif'} (brightfield | fluorescence, "
              f"every {g['keep_stride']}th frame, 2 um pixels)")
    except ImportError:
        print("  (no GIF: imageio not installed)")

    # How dilute must the stream be? Same sort, same cells, more of them in view.
    sweep = pd.DataFrame(parallel_map(
        crowding_case, [(params, film.outcome, n) for n in (3, 8, 20)]))
    print("\nthe same 120 cells filmed at increasing density:")
    print(sweep.round(1).to_string(index=False))
    sweep.to_csv(ASSETS / "13_crowding.csv", index=False)


if __name__ == "__main__":
    main()
