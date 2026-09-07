"""``biosim`` command-line interface.

Commands
--------
``biosim list``                  installed instruments and solver back-ends
``biosim doctor``                environment diagnosis: what works, what is missing
``biosim init <instrument>``     write a working example config
``biosim run <config.yaml>``     run an experiment and save results
``biosim sweep <config.yaml>``   parameter sweep, writes a long table + NetCDF
``biosim dashboard <target>``    serve the interactive dashboard in a browser
``biosim materials``            the material library with its DOIs and ASSUMPTIONs
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from biosim_lab import __version__

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="biosim-lab — open-source virtual laboratory instruments.",
)
console = Console()


def _instruments() -> dict[str, Any]:
    """Installed instruments, falling back to the built-ins in a source checkout."""
    from biosim_lab.registry import installed_instruments

    return installed_instruments()


@app.command("list")
def list_plugins() -> None:
    """List installed instrument plugins and solver back-ends."""
    from biosim_lab.core.solver import solver_report

    table = Table(title="Instruments", header_style="bold")
    table.add_column("name")
    table.add_column("display name")
    table.add_column("description", overflow="fold")
    for name, cls in sorted(_instruments().items()):
        table.add_row(name, cls.display_name, cls.description)
    console.print(table)

    table = Table(title="Solver back-ends", header_style="bold")
    table.add_column("name")
    table.add_column("kind")
    table.add_column("available")
    table.add_column("detail", overflow="fold")
    for row in solver_report():
        table.add_row(
            row["name"],
            row["kind"],
            "[green]yes[/green]" if row["available"] else "[yellow]no[/yellow]",
            row["detail"],
        )
    console.print(table)


@app.command()
def doctor() -> None:
    """Report which back-ends are installed and which features they unlock."""
    from biosim_lab.core.geometry import gmsh_available
    from biosim_lab.core.plugin import optional_import
    from biosim_lab.core.solver import solver_report
    from biosim_lab.core.viz.napari_layers import napari_available
    from biosim_lab.instruments.cell_counter.segmentation import available_backends

    console.print(f"[bold]biosim-lab {__version__}[/bold]")

    table = Table(title="Environment", header_style="bold")
    table.add_column("component")
    table.add_column("status")
    table.add_column("detail / what it unlocks", overflow="fold")

    def add(name: str, ok: bool, detail: str) -> None:
        table.add_row(name, "[green]OK[/green]" if ok else "[yellow]missing[/yellow]", detail)

    for module, unlocks in (
        ("numpy", "core"),
        ("scipy", "core"),
        ("xarray", "core"),
        ("pandas", "core"),
        ("pydantic", "config validation"),
        ("pint", "unit checking"),
        ("skfem", "built-in FEM solvers"),
        ("meshio", "Gmsh mesh import (optional: biosim-lab[mesh])"),
        ("plotly", "figures"),
        ("panel", "dashboards"),
        ("pyvista", "3-D fields and trajectory movies"),
        ("skimage", "segmentation"),
        ("trackpy", "track linking"),
        ("h5netcdf", "NetCDF output"),
        ("pyarrow", "Parquet output"),
    ):
        mod = optional_import(module)
        add(module, mod is not None, unlocks if mod is not None
            else f"{unlocks} unavailable — pip install {module}")

    ok, why = gmsh_available()
    add("gmsh", ok, why if ok else why)

    ok, why = napari_available()
    add("napari", ok, why if ok else f"{why}; instruments run and export without it")

    for name, (ok, detail) in available_backends().items():
        add(f"segmentation:{name}", ok, detail)

    for row in solver_report():
        add(f"solver:{row['name']}", row["available"],
            f"{row['kind']} — {row['detail']}")

    console.print(table)

    from biosim_lab.registry import discovery_source

    instruments = _instruments()
    console.print(
        f"\n[bold]{len(instruments)} instrument(s) discovered:[/bold] "
        + ", ".join(sorted(instruments))
        + f"\n[dim]found via {discovery_source()}[/dim]"
    )
    if not any(r["available"] and r["name"] in ("openfoam", "elmer")
               for r in solver_report()):
        console.print(
            "\n[yellow]Heavy solver back-ends are not installed.[/yellow] "
            "Acoustic streaming is disabled and the analytic Rayleigh approximation "
            "is used instead; the piezoelectric IDT solution is replaced by the "
            "documented voltage-to-pressure calibration. Stages 1-3 are complete "
            "without them."
        )


def _get_instrument(name: str) -> Any:
    """Look up one instrument, tolerating an uninstalled checkout."""
    instruments = _instruments()
    if name not in instruments:
        available = ", ".join(sorted(instruments)) or "none"
        raise KeyError(f"unknown instrument {name!r}; installed: {available}")
    return instruments[name]


@app.command()
def init(
    instrument: str = typer.Argument(..., help="instrument name, e.g. saw_sorter"),
    output: Path = typer.Option(Path("config.yaml"), "--output", "-o"),
) -> None:
    """Write a working example configuration for INSTRUMENT."""
    from biosim_lab.core.config import ExperimentConfig

    cls = _get_instrument(instrument)
    cfg = ExperimentConfig.model_validate(cls.example_config())
    path = cfg.to_yaml(output)
    console.print(f"[green]wrote[/green] {path}")


@app.command()
def run(
    config: Path = typer.Argument(..., exists=True, help="experiment YAML"),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-o"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
    replicates: int = typer.Option(
        1, "--replicates", "-n",
        help="run this many times with different random samples and report the "
             "spread as well as the point estimate",
    ),
) -> None:
    """Run an experiment described by CONFIG and save the results."""
    from biosim_lab.core.config import ExperimentConfig
    from biosim_lab.core.io import save_result

    cfg = ExperimentConfig.from_yaml(config)
    if output_dir is not None:
        cfg = cfg.model_copy(update={"output_dir": output_dir})

    cls = _get_instrument(cfg.instrument)
    instrument = cls(cfg)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        instrument.setup()
        result = instrument.run()

    written = save_result(result, cfg.output_dir, cfg.name)

    replicate_summary = None
    if replicates > 1:
        if not hasattr(instrument, "replicate"):
            console.print(
                f"[yellow]{cfg.instrument} does not support replicates; "
                "reporting the single run.[/yellow]"
            )
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                replicate_summary = instrument.replicate(n_replicates=replicates)
            table_path = Path(cfg.output_dir) / f"{cfg.name}_replicates.csv"
            replicate_summary["table"].to_csv(table_path, index=False)
            written["replicates"] = table_path

    if not quiet:
        table = Table(title=f"{cfg.name} — {cls.display_name}", header_style="bold")
        table.add_column("metric")
        table.add_column("value", justify="right")
        for key, value in result.metrics.items():
            if isinstance(value, dict):
                continue
            table.add_row(key, f"{value:.4g}" if isinstance(value, float) else str(value))
        console.print(table)

        if replicate_summary is not None:
            spread = Table(
                title=f"across {replicates} replicates (different cell samples)",
                header_style="bold",
            )
            spread.add_column("metric")
            for column in ("mean", "std", "95 % CI"):
                spread.add_column(column, justify="right")
            for name, summary in replicate_summary["summary"].items():
                spread.add_row(
                    name, f"{summary.mean:.3g}", f"{summary.std:.3g}",
                    f"[{summary.low:.3g}, {summary.high:.3g}]",
                )
            console.print(spread)

        for w in caught:
            if issubclass(w.category, UserWarning):
                console.print(f"[yellow]{w.category.__name__}:[/yellow] {w.message}")

        for kind, path in written.items():
            console.print(f"[green]wrote[/green] {kind}: {path}")


@app.command()
def sweep(
    config: Path = typer.Argument(..., exists=True),
    param: list[str] = typer.Option(
        ...,
        "--param",
        "-p",
        help="NAME=v1,v2,v3 — repeat for a multi-dimensional grid "
        "(values may carry units, e.g. -p 'frequency=5 MHz,10 MHz')",
    ),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-o"),
) -> None:
    """Sweep parameters around the operating point in CONFIG."""
    from biosim_lab.core.config import ExperimentConfig
    from biosim_lab.core.io import save_dataset, save_table

    cfg = ExperimentConfig.from_yaml(config)
    out_dir = output_dir or cfg.output_dir
    cls = _get_instrument(cfg.instrument)
    instrument = cls(cfg)
    if not hasattr(instrument, "sweep"):
        raise typer.BadParameter(f"{cfg.instrument} does not support sweeps")

    grid: dict[str, list[str]] = {}
    for item in param:
        if "=" not in item:
            raise typer.BadParameter(f"expected NAME=v1,v2,... got {item!r}")
        name, values = item.split("=", 1)
        grid[name.strip()] = [v.strip() for v in values.split(",")]

    console.print(f"sweeping {', '.join(f'{k} ({len(v)})' for k, v in grid.items())} …")
    df, ds = instrument.sweep(grid, progress=True)  # type: ignore[attr-defined]

    table_path = save_table(df, Path(out_dir) / f"{cfg.name}_sweep.parquet")
    nc_path = save_dataset(ds, Path(out_dir) / f"{cfg.name}_sweep.nc")
    csv_path = Path(out_dir) / f"{cfg.name}_sweep.csv"
    df.to_csv(csv_path, index=False)
    for kind, path in (("table", table_path), ("netcdf", nc_path), ("csv", csv_path)):
        console.print(f"[green]wrote[/green] {kind}: {path}")


@app.command()
def dashboard(
    target: Path = typer.Argument(..., exists=True, help="experiment YAML or results .nc"),
    port: int = typer.Option(5006, "--port"),
    show: bool = typer.Option(True, "--show/--no-show", help="open a browser"),
    address: str = typer.Option("0.0.0.0", "--address"),
) -> None:
    """Serve the interactive dashboard for a config or a results file."""
    from biosim_lab.core.config import ExperimentConfig
    from biosim_lab.core.viz.dashboard import serve

    if target.suffix == ".nc":
        from biosim_lab.core.io import load_result

        result = load_result(target)
        instrument_name = json.loads(result.meta and json.dumps(result.meta) or "{}").get(
            "instrument", ""
        )
        if not instrument_name:
            raise typer.BadParameter(
                f"{target} does not record which instrument produced it; "
                "serve the experiment YAML instead"
            )
        raise typer.BadParameter(
            "serving a saved .nc re-runs the instrument; pass the experiment YAML "
            f"for {instrument_name} instead"
        )

    cfg = ExperimentConfig.from_yaml(target)
    cls = _get_instrument(cfg.instrument)
    instrument = cls(cfg)
    instrument.setup()
    if cls.name != "saw_sorter":
        # Dashboards that visualise a completed run need one; the sorter builds
        # its own simulations from the sliders.
        instrument.run()

    console.print(f"serving {cls.display_name} on http://localhost:{port} …")
    serve(instrument.dashboard(), port=port, show=show, address=address)


@app.command()
def materials(
    assumptions_only: bool = typer.Option(
        True, "--assumptions-only/--all", help="show only the ASSUMPTION-flagged values"
    ),
) -> None:
    """List material properties with their DOI or ASSUMPTION provenance."""
    from biosim_lab.core.materials import all_values, audit

    if assumptions_only:
        rows = audit(only_assumptions=True)
        total = len(all_values())
        table = Table(
            title=f"ASSUMPTION-flagged values ({len(rows)} of {total})", header_style="bold"
        )
        table.add_column("group")
        table.add_column("material")
        table.add_column("property")
        table.add_column("value", justify="right")
        table.add_column("why", overflow="fold")
        for row in rows:
            table.add_row(
                row["group"], row["material"], row["property"], row["value"],
                row["provenance"].removeprefix("ASSUMPTION: "),
            )
        console.print(table)
        console.print(
            "\nEvery other value carries a DOI. Run with --all to see them."
        )
        return

    table = Table(title="Material library", header_style="bold")
    table.add_column("group")
    table.add_column("material")
    table.add_column("property")
    table.add_column("value", justify="right")
    table.add_column("provenance", overflow="fold")
    for group, key, prop, value in all_values():
        table.add_row(group, key, prop, f"{value.magnitude:g} {value.unit}", str(value.prov))
    console.print(table)


@app.command()
def version() -> None:
    """Print the version."""
    console.print(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
