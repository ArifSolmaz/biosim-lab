"""Persistence and raw-instrument-data readers.

Simulation output
-----------------
Fields go to **NetCDF** (via ``xarray``), per-object tables to **Parquet**, and
a human-readable **CSV** summary sits next to them.  :func:`save_result` writes
all three under one stem so a run is one directory listing.

Raw instrument data
-------------------
Readers for the file formats real instruments export, so the same analysis code
runs on measured data:

* :func:`read_rtca_csv` — xCELLigence RTCA Cell Index exports (CSV / XLSX).
* :func:`read_image_stack` — multi-page TIFF or a directory of images.
* :func:`read_generic_timeseries` — long or wide CSV with a time column.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from biosim_lab.core.plugin import InstrumentResult, optional_import

# ---------------------------------------------------------------------------
# simulation results
# ---------------------------------------------------------------------------


def _netcdf_engine() -> str:
    """Pick whichever NetCDF back-end is actually usable.

    ``h5netcdf`` imports fine without ``h5py`` but fails at write time, so its
    real dependency is checked here rather than discovered mid-run.
    """
    candidates = (
        ("netcdf4", ("netCDF4",)),
        ("h5netcdf", ("h5netcdf", "h5py")),
        ("scipy", ("scipy",)),
    )
    for engine, modules in candidates:
        if all(optional_import(m) is not None for m in modules):
            return engine
    raise RuntimeError(
        "no usable NetCDF back-end; install one of netCDF4, h5netcdf+h5py, or scipy"
    )


def _sanitise_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Coerce a Dataset into something NetCDF can actually store.

    Two things NetCDF rejects and xarray does not catch for us:

    * attributes that are not strings or numbers — JSON-encoded here;
    * ``bool`` — neither as an attribute nor as a variable dtype, since the
      classic NetCDF type list has no boolean. Booleans become ``int8``.
    """
    ds = ds.copy()
    for name, var in list(ds.data_vars.items()):
        if var.dtype == bool:
            ds[name] = var.astype("i1")
            ds[name].attrs["dtype"] = "bool"
    for obj in [ds, *ds.variables.values()]:
        for key, value in list(obj.attrs.items()):
            if isinstance(value, (bool, np.bool_)):
                obj.attrs[key] = int(value)
            elif isinstance(value, (str, int, float, np.number, np.ndarray, list, tuple)):
                continue
            else:
                obj.attrs[key] = json.dumps(value, default=str)
    return ds


def save_dataset(ds: xr.Dataset, path: str | Path) -> Path:
    """Write an :class:`xarray.Dataset` to NetCDF, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _sanitise_attrs(ds).to_netcdf(path, engine=_netcdf_engine())
    return path


def load_dataset(path: str | Path) -> xr.Dataset:
    """Read a NetCDF file written by :func:`save_dataset`."""
    return xr.load_dataset(path, engine=_netcdf_engine())


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a table to Parquet, falling back to CSV when pyarrow is absent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if optional_import("pyarrow") is None and optional_import("fastparquet") is None:
        csv = path.with_suffix(".csv")
        df.to_csv(csv, index=False)
        return csv
    df.to_parquet(path, index=False)
    return path


def load_table(path: str | Path) -> pd.DataFrame:
    """Read a table written by :func:`save_table` (Parquet or CSV)."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def save_result(result: InstrumentResult, output_dir: str | Path, stem: str) -> dict[str, Path]:
    """Persist an :class:`InstrumentResult` as ``<stem>.nc`` / ``.parquet`` / ``_metrics.csv``.

    Returns
    -------
    dict
        ``{"fields": path, "table": path | None, "metrics": path}``
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    written["fields"] = save_dataset(result.to_dataset(), output_dir / f"{stem}.nc")

    if result.table is not None and len(result.table):
        written["table"] = save_table(result.table, output_dir / f"{stem}.parquet")

    metrics_path = output_dir / f"{stem}_metrics.csv"
    flat = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
            for k, v in result.metrics.items()}
    pd.DataFrame([flat]).to_csv(metrics_path, index=False)
    written["metrics"] = metrics_path
    return written


def load_result(path: str | Path) -> InstrumentResult:
    """Load an :class:`InstrumentResult` previously written by :func:`save_result`."""
    path = Path(path)
    ds = load_dataset(path)
    table_path = path.with_suffix(".parquet")
    table = None
    if table_path.exists():
        table = load_table(table_path)
    elif path.with_suffix(".csv").exists():
        table = load_table(path.with_suffix(".csv"))
    return InstrumentResult.from_dataset(ds, table=table)


# ---------------------------------------------------------------------------
# raw instrument readers
# ---------------------------------------------------------------------------

_WELL_RE = re.compile(r"^([A-P])(\d{1,2})$", re.IGNORECASE)


def _read_ragged_csv(path: Path) -> pd.DataFrame:
    """Read a delimited file whose rows have *different* field counts.

    Instrument exports routinely put a few short metadata lines above the real
    header, which makes ``pandas.read_csv`` raise on the field-count mismatch.
    Rows are therefore split by hand and padded to the widest row.
    """
    import csv as _csv

    text = path.read_text(encoding="utf-8-sig", errors="replace")
    sample = "\n".join(text.splitlines()[:20])
    try:
        delimiter = _csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except _csv.Error:
        delimiter = ","
    rows = [row for row in _csv.reader(text.splitlines(), delimiter=delimiter) if row]
    if not rows:
        raise ValueError(f"{path}: file is empty")
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]
    return pd.DataFrame(padded)


def _is_well(label: Any) -> bool:
    return bool(_WELL_RE.match(str(label).strip()))


def read_rtca_csv(path: str | Path, *, time_unit: str = "hour") -> xr.Dataset:
    """Read an xCELLigence RTCA Cell Index export (CSV or XLSX).

    The RTCA software exports a wide table: one time column (elapsed hours) and
    one column per well, labelled ``A1`` ... ``H12``.  Some export profiles put
    a few metadata lines above the header, so the header row is located by
    looking for the first row whose cells parse as well labels.

    Returns
    -------
    xarray.Dataset
        ``cell_index(time, well)`` with ``time`` in seconds.
    """
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path, header=None)
    else:
        raw = _read_ragged_csv(path)

    header_row = None
    for i in range(min(len(raw), 30)):
        row = raw.iloc[i].astype(str).str.strip()
        if (row.map(_is_well)).sum() >= 2:
            header_row = i
            break
    if header_row is None:
        raise ValueError(
            f"{path}: no header row with well labels (A1, B2, ...) found in the first 30 rows"
        )

    header = raw.iloc[header_row].astype(str).str.strip().tolist()
    body = raw.iloc[header_row + 1:].reset_index(drop=True)
    body.columns = header
    body = body.dropna(how="all")

    well_cols = [c for c in body.columns if _is_well(c)]
    time_col = next((c for c in body.columns if not _is_well(c)
                     and re.search(r"time|hour|saat", str(c), re.IGNORECASE)), None)
    if time_col is None:
        non_well = [c for c in body.columns if not _is_well(c)]
        if not non_well:
            raise ValueError(f"{path}: no time column found")
        time_col = non_well[0]

    time = pd.to_numeric(body[time_col], errors="coerce").to_numpy(dtype=float)
    scale = {"hour": 3600.0, "minute": 60.0, "second": 1.0}[time_unit]
    values = body[well_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    keep = ~np.isnan(time)

    ds = xr.Dataset(
        {"cell_index": (("time", "well"), values[keep])},
        coords={"time": time[keep] * scale, "well": [str(w).upper() for w in well_cols]},
    )
    ds["time"].attrs["units"] = "s"
    ds["cell_index"].attrs["units"] = "dimensionless"
    ds.attrs["source_file"] = str(path)
    ds.attrs["reader"] = "read_rtca_csv"
    return ds


def read_image_stack(path: str | Path, *, pattern: str = "*.tif*") -> np.ndarray:
    """Read a multi-page TIFF, or every image in a directory, as ``(t, y, x[, c])``.

    Uses ``imageio`` when available and falls back to ``tifffile`` /
    ``skimage.io``; raises a clear error if none is installed.
    """
    path = Path(path)
    iio = optional_import("imageio.v3")
    if path.is_dir():
        files = sorted(path.glob(pattern))
        if not files:
            raise FileNotFoundError(f"{path}: no files matching {pattern!r}")
        if iio is None:
            raise RuntimeError("reading image sequences needs imageio")
        return np.stack([np.asarray(iio.imread(f)) for f in files])

    if iio is not None:
        arr = np.asarray(iio.imread(path))
    else:
        tifffile = optional_import("tifffile")
        if tifffile is None:
            raise RuntimeError("reading TIFF stacks needs imageio or tifffile")
        arr = np.asarray(tifffile.imread(path))
    return arr if arr.ndim >= 3 else arr[np.newaxis]


def read_generic_timeseries(
    path: str | Path, *, time_column: str | None = None, time_unit: str = "second"
) -> pd.DataFrame:
    """Read a CSV/XLSX time series, normalising the time column to seconds."""
    path = Path(path)
    df = (pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"}
          else pd.read_csv(path, sep=None, engine="python"))
    if time_column is None:
        time_column = next(
            (c for c in df.columns if re.search(r"time|hour|min|sec|saat", str(c), re.I)),
            df.columns[0],
        )
    scale = {"hour": 3600.0, "minute": 60.0, "second": 1.0}[time_unit]
    df = df.rename(columns={time_column: "time"})
    df["time"] = pd.to_numeric(df["time"], errors="coerce") * scale
    return df.dropna(subset=["time"]).reset_index(drop=True)


__all__ = [
    "save_dataset",
    "load_dataset",
    "save_table",
    "load_table",
    "save_result",
    "load_result",
    "read_rtca_csv",
    "read_image_stack",
    "read_generic_timeseries",
]
