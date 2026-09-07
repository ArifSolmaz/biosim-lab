"""Persistence and raw-instrument readers."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from biosim_lab.core.io import (
    load_dataset,
    load_result,
    read_generic_timeseries,
    read_image_stack,
    read_rtca_csv,
    save_dataset,
    save_result,
)
from biosim_lab.core.plugin import InstrumentResult


def test_dataset_roundtrip(tmp_path):
    ds = xr.Dataset({"p": ("x", np.arange(5.0))}, coords={"x": np.arange(5)})
    ds.attrs["note"] = "hello"
    path = save_dataset(ds, tmp_path / "sub" / "f.nc")
    assert path.exists()
    back = load_dataset(path)
    assert np.allclose(back["p"].values, ds["p"].values)
    assert back.attrs["note"] == "hello"


def test_nonstring_attributes_are_json_encoded(tmp_path):
    ds = xr.Dataset({"p": ("x", np.arange(3.0))})
    ds.attrs["nested"] = {"a": 1, "b": [1, 2]}
    back = load_dataset(save_dataset(ds, tmp_path / "f.nc"))
    assert isinstance(back.attrs["nested"], str)


def test_result_roundtrip_preserves_metrics_and_table(tmp_path):
    result = InstrumentResult(
        fields=xr.Dataset({"y": ("t", np.arange(4.0))}),
        metrics={"efficiency_percent": 99.5, "nested": {"a": 1}},
        table=pd.DataFrame({"cell": [1, 2], "x": [0.1, 0.2]}),
        meta={"instrument": "test"},
    )
    written = save_result(result, tmp_path, "run")
    assert written["fields"].exists() and written["metrics"].exists()
    back = load_result(written["fields"])
    assert back.metrics["efficiency_percent"] == pytest.approx(99.5)
    assert back.meta["instrument"] == "test"
    assert list(back.table["cell"]) == [1, 2]


def _write_rtca_csv(path, extra_header_rows=2):
    lines = []
    for i in range(extra_header_rows):
        lines.append(f"RTCA Software Export,row {i}")
    lines.append("Time (h),A1,A2,B1")
    for t in range(5):
        lines.append(f"{t * 0.5},{1.0 + t},{2.0 + t},{0.5 * t}")
    path.write_text("\n".join(lines), encoding="utf-8")


def test_rtca_reader_skips_metadata_rows_and_converts_time(tmp_path):
    path = tmp_path / "export.csv"
    _write_rtca_csv(path)
    ds = read_rtca_csv(path)
    assert list(ds["well"].values) == ["A1", "A2", "B1"]
    assert ds["time"].values[1] == pytest.approx(0.5 * 3600)
    assert ds["cell_index"].shape == (5, 3)
    assert ds["cell_index"].values[0, 0] == pytest.approx(1.0)


def test_rtca_reader_reports_a_missing_header(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("no,well,labels,here\n1,2,3,4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="well labels"):
        read_rtca_csv(path)


def test_generic_timeseries_normalises_time(tmp_path):
    path = tmp_path / "ts.csv"
    path.write_text("time_min,signal\n0,1\n10,2\n20,3\n", encoding="utf-8")
    df = read_generic_timeseries(path, time_unit="minute")
    assert list(df["time"]) == [0.0, 600.0, 1200.0]


def test_image_stack_reader_handles_a_directory(tmp_path):
    import imageio.v3 as iio

    for i in range(3):
        iio.imwrite(tmp_path / f"f{i:02d}.tif", (np.ones((8, 8)) * i).astype(np.uint8))
    stack = read_image_stack(tmp_path)
    assert stack.shape == (3, 8, 8)
    with pytest.raises(FileNotFoundError):
        read_image_stack(tmp_path, pattern="*.png")
