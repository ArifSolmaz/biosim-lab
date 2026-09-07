"""Stage 3: synthetic imaging, counting and tracking."""

import numpy as np
import pytest

from biosim_lab.core.config import ExperimentConfig
from biosim_lab.instruments.cell_counter import CellCounter
from biosim_lab.instruments.cell_counter.counting import (
    NEUBAUER_DEPTH_M,
    concentration_from_field,
    counting_uncertainty,
    field_volume_ml,
)
from biosim_lab.instruments.cell_counter.segmentation import available_backends, segment
from biosim_lab.instruments.cell_counter.synthetic import (
    SyntheticImageSpec,
    synthetic_field,
    synthetic_movie,
)
from biosim_lab.instruments.cell_tracker import CellTracker
from biosim_lab.instruments.cell_tracker.tracking import fit_msd


def test_synthetic_field_has_the_requested_ground_truth():
    field = synthetic_field(SyntheticImageSpec(n_cells=50, shape=(256, 256), seed=1))
    assert field["image"].shape == (256, 256)
    assert field["image"].min() >= 0 and field["image"].max() <= 1
    assert len(field["truth"]) == 50
    assert field["labels"].max() == 50


def test_classical_backend_is_always_available():
    ok, _ = available_backends()["classical"]
    assert ok


def test_segmentation_recovers_most_of_the_ground_truth():
    field = synthetic_field(
        SyntheticImageSpec(n_cells=80, shape=(384, 384), allow_touching=False, seed=5)
    )
    result = segment(field["image"], backend="classical")
    recall = result.n_objects / len(field["truth"])
    assert 0.8 < recall <= 1.15, f"recall {recall:.2f}"
    assert {"area", "equivalent_diameter", "mean_intensity"} <= set(
        result.properties.columns
    )


def test_segmentation_of_an_empty_field_finds_nothing():
    blank = np.full((128, 128), 0.25, dtype=np.float32)
    result = segment(blank, backend="classical", min_radius_px=6)
    assert result.n_objects <= 2  # noise may leave a speck; a whole field must not


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        segment(np.zeros((16, 16)), backend="nope")


def test_missing_optional_backends_explain_the_extra():
    from biosim_lab.instruments.cell_counter.segmentation import segment_cellpose

    ok, _ = available_backends()["cellpose"]
    if ok:  # pragma: no cover
        pytest.skip("cellpose installed here")
    with pytest.raises(RuntimeError, match=r"biosim-lab\[segmentation\]"):
        segment_cellpose(np.zeros((16, 16)))


def test_neubauer_volume_and_the_ten_thousand_rule():
    """One 1 mm x 1 mm x 0.1 mm square is 1e-4 mL, hence the classic 1e4 factor."""
    volume = field_volume_ml((1000, 1000), 1e-6, NEUBAUER_DEPTH_M)
    assert volume == pytest.approx(1e-4)
    conc, _ = concentration_from_field(1, (1000, 1000), 1e-6)
    assert conc == pytest.approx(1e4)


def test_dilution_factor_scales_the_concentration():
    a, _ = concentration_from_field(100, (512, 512), 0.65e-6, dilution_factor=1.0)
    b, _ = concentration_from_field(100, (512, 512), 0.65e-6, dilution_factor=2.0)
    assert b == pytest.approx(2 * a)


def test_counting_uncertainty_is_poisson():
    assert counting_uncertainty(100) == pytest.approx(0.1)
    assert counting_uncertainty(0) == float("inf")


def test_counter_instrument_end_to_end():
    cfg = ExperimentConfig.model_validate(CellCounter.example_config())
    result = CellCounter(cfg).run()
    m = result.metrics
    assert m["n_total"] > 50
    assert 0 <= m["viability_percent"] <= 100
    assert m["concentration_per_ml"] > 0
    assert m["detection_recall"] > 0.7
    assert 10 < m["mean_diameter_um"] < 25
    assert "labels" in result.fields


def test_synthetic_movie_tracks_are_continuous():
    data = synthetic_movie(
        SyntheticImageSpec(shape=(192, 192), n_cells=20, seed=3), n_frames=8
    )
    assert data["movie"].shape == (8, 192, 192)
    truth = data["truth"]
    assert np.unique(truth["particle"]).size == 20
    assert np.unique(truth["frame"]).size == 8


def test_tracker_instrument_recovers_the_planted_speed():
    cfg = ExperimentConfig.model_validate(CellTracker.example_config())
    result = CellTracker(cfg).run()
    m = result.metrics
    assert m["n_tracks"] > 0.5 * m["ground_truth_n_tracks"]
    assert m["mean_speed_um_per_min"] == pytest.approx(
        m["ground_truth_speed_um_per_min"], rel=0.3
    )
    assert m["median_persistence"] > 0.5, "persistent motion must not look diffusive"
    assert m["msd_alpha"] > 1.4, "persistent random walk is superdiffusive"


def test_msd_fit_identifies_pure_diffusion():
    import pandas as pd

    lag = np.logspace(0, 2, 20)
    msd = pd.DataFrame({"lag_s": lag, "msd_m2": 4 * 1e-12 * lag, "n_samples": 10})
    alpha, d = fit_msd(msd)
    assert alpha == pytest.approx(1.0, abs=1e-6)
    assert d == pytest.approx(1e-12, rel=1e-6)


def test_msd_fit_identifies_ballistic_motion():
    import pandas as pd

    lag = np.logspace(0, 2, 20)
    msd = pd.DataFrame({"lag_s": lag, "msd_m2": 1e-12 * lag**2, "n_samples": 10})
    alpha, _ = fit_msd(msd)
    assert alpha == pytest.approx(2.0, abs=1e-6)
