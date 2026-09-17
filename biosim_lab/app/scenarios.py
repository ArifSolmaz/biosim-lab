"""Named operating points for the sorter page.

Twenty-odd controls with no starting point is a blank page, and the interesting
operating points of this device are already written down --- in ``configs/`` and
in :mod:`biosim_lab.instruments.saw_sorter.protocols`. This module reads them
from those files rather than restating their numbers, so a scenario cannot
drift from the configuration the command line would run.

What a scenario cannot always express: the sidebar carries one target and one
background population, while a blood background has three or four. Where a
source is reduced to fit, :func:`scenario` says so in ``notes`` and the page
prints it --- a preset that silently dropped two thirds of the sample would be
worse than no preset at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

UL_MIN = 1e-9 / 60.0  # m^3/s per uL/min

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

#: Display name -> where to read it from. ``None`` leaves every widget alone.
SOURCES: dict[str, tuple[str, str] | None] = {
    "Default — single node, 300 µm channel": None,
    "MCF-7 from erythrocytes": ("config", "ctc_vs_rbc.yaml"),
    "CTC assay — 1 % spike into blood": ("config", "saw_sorter_whole_blood.yaml"),
    "20 MHz specification point (multi-node)": ("config", "saw_sorter_20mhz_spec.yaml"),
    "Same device, solved by FEM": ("config", "saw_sorter_fem.yaml"),
    "Li et al. 2015 — tilted-angle CTC sorter": ("protocol", "li_2015_tassaw"),
}


def _snap(value: float, lo: float, hi: float, step: float | None = None) -> float:
    """Clamp into a widget's range, and onto its step if it has one."""
    v = min(max(float(value), lo), hi)
    if step:
        v = lo + round((v - lo) / step) * step
        v = min(max(v, lo), hi)
    return round(v, 6)


def _from_params(params: Any, seed: int | None) -> tuple[dict[str, Any], list[str]]:
    """Map a validated ``SAWSorterParams`` onto the sorter page's widget keys."""
    notes: list[str] = []
    widgets: dict[str, Any] = {}

    widgets["sorter_frequency_mhz"] = _snap(params.frequency / 1e6, 1.0, 40.0, 0.001)
    if params.voltage_pp is None:
        notes.append(
            "the source drives the device by pressure amplitude rather than voltage, "
            "so the voltage slider is left where it was"
        )
    else:
        widgets["sorter_voltage_pp"] = _snap(params.voltage_pp, 1.0, 40.0, 0.5)
    widgets["sorter_flow_ul_min"] = _snap(params.flow_rate / UL_MIN, 0.5, 60.0, 0.5)
    widgets["sorter_width_um"] = _snap(params.channel_width * 1e6, 100.0, 800.0, 10.0)
    widgets["sorter_height_um"] = _snap(params.channel_height * 1e6, 20.0, 200.0, 5.0)
    widgets["sorter_length_mm"] = _snap(params.channel_length * 1e3, 0.2, 10.0, 0.1)
    widgets["sorter_temperature_c"] = _snap(params.temperature - 273.15, 4.0, 45.0, 0.5)
    widgets["sorter_inlet_viability"] = _snap(params.inlet_viability, 0.5, 1.0, 0.01)
    widgets["sorter_tilt_angle_deg"] = _snap(params.tilt_angle_deg, -89.0, 89.0)
    widgets["sorter_inlet"] = params.inlet
    if params.inlet == "side":
        widgets["sorter_inlet_side"] = params.inlet_side

    # A tilt forbids the cross-section FEM (the model refuses the pair), and the
    # page does not offer it there, so never seed a state the widget cannot hold.
    field_model = getattr(params, "field_model", "analytic")
    widgets["sorter_mode"] = "analytic" if widgets["sorter_tilt_angle_deg"] else field_model
    if field_model == "fem" and widgets["sorter_mode"] == "analytic":
        notes.append("the source asks for FEM, which a tilted device cannot use")
    resolution = getattr(params, "fem_resolution", None)
    if resolution and widgets["sorter_mode"] == "fem":
        widgets["sorter_fem_resolution"] = int(_snap(resolution, 16, 64, 8))

    # Read the layout the source asked for; only then the control that layout
    # actually shows, so a tilted device arrives with its divider, not with a
    # centre band it never used.
    layout = getattr(params, "outlet_layout", "centre_band")
    widgets["sorter_outlet_layout"] = layout
    if layout == "centre_band":
        widgets["sorter_collection_fraction"] = _snap(
            getattr(params, "collection_fraction", 1 / 3), 0.05, 0.9, 0.01)
    else:
        widgets["sorter_split_position"] = _snap(
            getattr(params, "split_position", 0.5), 0.05, 0.95, 0.01)
        widgets["sorter_collect_side"] = getattr(params, "collect_side", "right")

    targets = [p for p in params.populations if p.target]
    others = sorted((p for p in params.populations if not p.target),
                    key=lambda p: p.count, reverse=True)
    if targets:
        widgets["sorter_target"] = targets[0].cell_type
        widgets["sorter_n_cells"] = int(_snap(targets[0].count, 50, 600, 50))
        if widgets["sorter_n_cells"] != targets[0].count:
            notes.append(
                f"the source runs {targets[0].count} target cells; the slider "
                f"holds {widgets['sorter_n_cells']}"
            )
    if others:
        widgets["sorter_background"] = others[0].cell_type
        if len(others) > 1:
            dropped = ", ".join(f"{p.count} {p.cell_type}" for p in others[1:])
            notes.append(
                f"the page carries one background population, so only "
                f"{others[0].cell_type} is kept here — {dropped} are not in the run"
            )
    if seed is not None:
        # The seed box holds six digits; a source may carry more (a date, say).
        # Fold it rather than drop it, and say so, because a different seed is
        # a different draw of cells even though the device is the same.
        folded = int(seed) % 1_000_000
        widgets["sorter_seed"] = folded
        if folded != int(seed):
            notes.append(
                f"the source's seed ({seed}) does not fit the six-digit seed box, "
                f"so {folded} is used — the same device, a different draw of cells"
            )
    return widgets, notes


def scenario(name: str) -> dict[str, Any] | None:
    """Widget values and caveats for a named scenario, or ``None`` for the default."""
    source = SOURCES.get(name)
    if source is None:
        return None
    kind, ref = source
    from biosim_lab.instruments.saw_sorter.simulate import SAWSorterParams

    if kind == "config":
        from biosim_lab.core.config import ExperimentConfig

        config = ExperimentConfig.from_yaml(CONFIG_DIR / ref)
        params = config.validated_params(SAWSorterParams)
        widgets, notes = _from_params(params, config.seed)
        return {"widgets": widgets, "notes": notes,
                "source": f"configs/{ref}", "description": config.description or ""}

    from biosim_lab.instruments.saw_sorter import protocols

    protocol = getattr(protocols, ref)()
    params = protocol["params"]
    if not isinstance(params, SAWSorterParams):
        params = SAWSorterParams(**params)
    widgets, notes = _from_params(params, getattr(params, "seed", None))
    reported = protocol.get("reported", {})
    description = ""
    if reported:
        description = (
            f"Published: {reported.get('cancer_recovery_percent', float('nan')):.0f} % "
            "cancer-cell recovery with "
            f"{reported.get('wbc_removal_percent', float('nan')):.0f} % of leukocytes "
            "removed."
        )
    return {"widgets": widgets, "notes": notes,
            "source": "biosim_lab.instruments.saw_sorter.protocols." + ref,
            "description": description}


__all__ = ["SOURCES", "scenario"]
