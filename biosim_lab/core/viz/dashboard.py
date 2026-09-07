"""Panel scaffolding shared by every instrument dashboard.

Instruments build their own controls, but they all get the same shell: a title
block, a metrics strip, a tabbed figure area and a data table.  Keeping that
here means a new plugin gets a usable dashboard in a few lines
(see CONTRIBUTING.md step 7).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd

from biosim_lab.core.plugin import optional_import
from biosim_lab.core.viz.theme import PALETTE


def require_panel() -> Any:
    """Import Panel and enable the extensions the platform uses."""
    pn = optional_import("panel")
    if pn is None:
        raise RuntimeError(
            "panel is required for dashboards; install it with `pip install biosim-lab`"
        )
    pn.extension("plotly", "tabulator", sizing_mode="stretch_width")
    return pn


def metric_tiles(metrics: dict[str, Any], keys: Sequence[str],
                 formats: dict[str, str] | None = None) -> Any:
    """A row of headline numbers.

    A stat tile is the right form when the data's job is to state one value —
    no axes, no chart furniture, just the number and its label.
    """
    pn = require_panel()
    formats = formats or {}
    tiles = []
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        fmt = formats.get(key, "{:.1f}" if isinstance(value, float) else "{}")
        try:
            text = fmt.format(value)
        except (ValueError, TypeError):
            text = str(value)
        tiles.append(
            pn.pane.HTML(
                f"""
                <div style="padding:14px 18px;border:1px solid {PALETTE['grid_light']};
                            border-radius:10px;background:{PALETTE['surface_light']};
                            min-width:150px">
                  <div style="font-size:11px;letter-spacing:.04em;text-transform:uppercase;
                              color:{PALETTE['text_secondary_light']}">
                    {key.replace('_', ' ')}
                  </div>
                  <div style="font-size:26px;font-weight:600;margin-top:4px;
                              color:{PALETTE['text_primary_light']}">{text}</div>
                </div>
                """,
                sizing_mode="fixed",
                width=190,
                height=86,
            )
        )
    return pn.Row(*tiles, sizing_mode="stretch_width")


def data_table(df: pd.DataFrame, *, height: int = 320, page_size: int = 20) -> Any:
    """A sortable, paged table — the accessible fallback for every figure."""
    pn = require_panel()
    return pn.widgets.Tabulator(
        df, height=height, page_size=page_size, pagination="local",
        show_index=False, sizing_mode="stretch_width",
    )


def shell(
    title: str,
    subtitle: str,
    *,
    controls: Iterable[Any] = (),
    tiles: Any | None = None,
    tabs: Sequence[tuple[str, Any]] = (),
    footer: str = "",
) -> Any:
    """Assemble the standard dashboard layout.

    Filters and controls live in a single row above the figures, as the
    interaction guidelines require, rather than scattered between them.
    """
    pn = require_panel()
    header = pn.pane.HTML(
        f"""
        <div style="padding:4px 0 10px 0">
          <div style="font-size:20px;font-weight:650;
                      color:{PALETTE['text_primary_light']}">{title}</div>
          <div style="font-size:13px;color:{PALETTE['text_secondary_light']};
                      margin-top:2px">{subtitle}</div>
        </div>
        """
    )
    control_row = pn.Row(*controls, sizing_mode="stretch_width") if controls else None
    parts: list[Any] = [header]
    if control_row is not None:
        parts.append(control_row)
    if tiles is not None:
        parts.append(tiles)
    if tabs:
        parts.append(pn.Tabs(*tabs, dynamic=False))
    if footer:
        parts.append(
            pn.pane.HTML(
                f"<div style='font-size:11px;color:{PALETTE['text_secondary_light']};"
                f"padding-top:10px'>{footer}</div>"
            )
        )
    return pn.Column(*parts, sizing_mode="stretch_width")


def serve(view: Any, *, port: int = 5006, show: bool = True, address: str = "0.0.0.0") -> Any:
    """Serve a dashboard on a local Bokeh server."""
    pn = require_panel()
    return pn.serve(view, port=port, show=show, address=address,
                    websocket_origin="*", title="biosim-lab")


__all__ = ["require_panel", "metric_tiles", "data_table", "shell", "serve"]
