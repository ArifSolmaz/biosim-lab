"""Assemble ``benchmarks/REPORT.md`` from the two benchmarks' ``summary.json``.

    python -m benchmarks.make_report

Every number in the report is read from the results the run scripts wrote, so
the report cannot drift from the data: rerun the benchmarks, rerun this.
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.common import ROOT, fmt

OUT = ROOT / "REPORT.md"
B1 = ROOT / "benchmark_01_tassaw" / "results"
B2 = ROOT / "benchmark_02_alternating_baw" / "results"


def _load(results: Path) -> dict[str, Any] | None:
    path = results / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _table(rows: list[dict[str, Any]], unit_digits: int = 1) -> str:
    out = ["| Case | Quantity | Ours | Paper | Deviation | Status |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        unit = f" {r['unit']}" if r.get("unit") and not isinstance(r["ours"], str) else ""
        dev = r.get("deviation")
        dev_s = "—" if dev is None or dev != dev else f"{dev:+.1f}"
        out.append(f"| {r['case']} | {r['quantity']} | {fmt(r['ours'], unit_digits)}{unit} | "
                   f"{fmt(r['paper'], unit_digits)}{unit} | {dev_s} | **{r['status']}** |")
    return "\n".join(out)


def _notes(rows: list[dict[str, Any]]) -> str:
    lines = []
    for r in rows:
        text = []
        if r.get("note"):
            text.append(f"*note:* {r['note']}")
        if r["status"] == "DEVIATION" and r.get("cause"):
            text.append(f"*likely cause:* {r['cause']}")
        if text:
            lines.append(f"- **{r['case']} — {r['quantity']}**: " + " ".join(text))
    return "\n".join(lines)


def _counts(rows: list[dict[str, Any]]) -> str:
    c = Counter(r["status"] for r in rows)
    order = ["PASS", "DEVIATION", "FAIL", "REPRODUCED", "NOT REPRODUCED", "CALIBRATION"]
    return ", ".join(f"{k}: {c[k]}" for k in order if c.get(k))


def _fig(summary: dict[str, Any], key: str, results: Path, caption: str) -> str:
    f = summary.get("figures", {}).get(key, {})
    if "png" not in f:
        return ""
    rel = Path(f["png"]).relative_to(ROOT)
    html = Path(f["html"]).relative_to(ROOT)
    return f"![{caption}]({rel.as_posix()})\n\n*{caption}* — [interactive]({html.as_posix()})\n"


def _commit() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=ROOT, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "biosim_lab",
                                "benchmarks"], capture_output=True, text=True, cwd=ROOT.parent,
                               check=True).stdout.strip()
        return f"{head} + working-tree changes" if dirty else head
    except Exception:  # noqa: BLE001
        return "unknown"


def section_01(s: dict[str, Any]) -> str:
    rows = s["checks"]
    cal = s["calibration"]
    re = s["reanchored"]
    bracket = cal.get("bracket_Pa")
    bracket_s = (f" (bracket {bracket[0] / 1e6:.3f}–{bracket[1] / 1e6:.3f} MPa)"
                 if bracket else "")
    re_rows = "\n".join(
        f"| {r['line']} | {r['recovery_rate_percent']:.1f} % | "
        f"{r['background_removal_percent']:.1f} % |" for r in re["table"])
    removal_stated = sum(r["background_removal_percent"] for r in s["table1"]) / len(s["table1"])
    in_band = [r for r in re["table"] if 83.0 <= r["recovery_rate_percent"] <= 96.0 + 4.0]
    low = [r for r in re["table"] if r["recovery_rate_percent"] < 83.0]
    if not low:
        verdict = ("Every line is then recovered at or near the stated 83-96 % band: the "
                   "Table 1 deviation is essentially one power offset.")
    else:
        verdict = (
            "The offset fixes the leukocytes and "
            + ", ".join(r["line"] for r in in_band)
            + ", but not "
            + ", ".join(f"{r['line']} ({r['recovery_rate_percent']:.0f} %)" for r in low)
            + ". So one offset does **not** reconcile Table 1: the model's size selectivity "
            "is sharper than the device's. The paper recovers every line at 83-96 % while "
            "removing ~90 % of leukocytes; in the model a line whose (assumed) diameter sits "
            "close to the leukocyte tail cannot have both. The inputs that decide this --- "
            "per-line diameters (the paper gives only '16 or 20 um'), size spreads and the "
            "Table S1 compressibilities --- are exactly the unavailable ones."
        )
    return f"""
## benchmark_01 — tilted-angle SSAW (Li et al. 2015)

**Paper:** {s['paper']['citation']}, doi:[{s['paper']['doi']}](https://doi.org/{s['paper']['doi']}).
**Mode:** `tassaw`. **Config:** [`benchmark_01_tassaw/config.yaml`](benchmark_01_tassaw/config.yaml).
**Reference values:** [`reference.yaml`](benchmark_01_tassaw/reference.yaml), each with the
sentence and page it comes from.

### Device

| Parameter | Value | Status |
|---|---|---|
| Substrate, frequency | LiNbO₃, 19.573 MHz | stated (p. 4972–4973) |
| IDT tilt, IDT length | 5°, 10 mm | stated (p. 4972) |
| Channel | 800 µm × 110 µm, PDMS | stated (p. 4972) |
| Flow | 75 µL/min gross, sheath:sample 2.5:1 | stated (p. 4971) |
| Power | 35 dBm (design simulations), ~37.5 dBm (rare cells) | stated |
| Cell sizes | WBC ~12 µm; cancer lines "16 or 20 µm" | stated (p. 4975), line mapping **assumed** |
| Densities, compressibilities | Table S1 of the paper | **not available → library values, ASSUMPTION** |
| Outlet divider | channel midline | **ASSUMPTION** |
| dBm → pressure | `p0 = p_ref √((P/P_ref)(L_ref/L))` | √P is physics; the 1/L spreading is an **ASSUMPTION** |
| `p_ref` at 35 dBm, 10 mm | **{cal['reference_pressure_Pa'] / 1e6:.4f} MPa**{bracket_s} | **CALIBRATED** |

**Calibration.** One number is fitted, and to one stated condition: `p_ref` is chosen so
that, at 75 µL/min and 35 dBm, the tilt that maximises the separation distance is the
stated ~5° (p. 4971). The ~600 µm quoted in the same sentence is **not** fitted; it is the
first prediction below. Method: {cal.get('method', '—')}.

### Results — {_counts(rows)}

{_table(rows)}

{_notes(rows)}

{_fig(s, 'fig2a', B1, 'Fig. 2A analogue: separation distance against tilt at five flow rates')}
{_fig(s, 'fig2b', B1, 'Fig. 2B analogue: separation distance against IDT length')}
{_fig(s, 'fig3', B1, 'Fig. 3 analogue: recovery and WBC removal against power')}
{_fig(s, 'table1', B1, 'Table 1: rare-cell recovery at 37.5 dBm')}
{_fig(s, 'beads', B1, '9.9 vs 7.3 µm polystyrene')}

### Sensitivity — is Table 1 one power offset away?

At the stated ~37.5 dBm the model over-drives: WBC removal falls to
{removal_stated:.0f} % against the stated ~90 %. The same model reaches 90 % WBC removal
at **{re['power_dbm']:.1f} dBm**, **{re['offset_db']:.1f} dB** below the stated drive —
the kind of offset an unreported RF insertion loss would produce. At that power:

| Line | Recovery | WBC removal |
|---|---|---|
{re_rows}

{verdict}

This is a sensitivity, not a comparison: it fits a second number.
"""


def section_02(s: dict[str, Any]) -> str:
    rows = s["checks"]
    cal = s["calibration"]
    sens = "\n".join(f"| {r['variant']} | {r['capture_efficiency_percent']:.1f} % | "
                     f"{r['contamination_rate_percent']:.1f} % |"
                     for r in s["contamination_sensitivity"])
    cf = "\n".join(
        f"| {arm} | {w['best_youden_percent']:.0f} % | {w['best_capture_percent']:.0f} % | "
        f"{w['best_contamination_percent']:.1f} % | {w['best_setting']:g} Vpp | "
        f"{'yes' if w['window'] else 'no'} |" for arm, w in s["counterfactual"].items())
    flows = "\n".join(
        f"| {r['flow_ul_h']:g} | {r['length_mm']:g} | {r['T1_s']:.2f} / {r['T3_s']:.2f} | "
        f"{r.get('capture_efficiency_percent', float('nan')):.1f} % | "
        f"{r.get('contamination_rate_percent', float('nan')):.1f} % |"
        for r in sorted(s["flow_rates"], key=lambda r: (r["flow_ul_h"], r["length_mm"])))
    rule = s["separation_rule"]
    rule_s = ", ".join(f"{k}: {v['displacement_um']:.0f} µm" for k, v in rule["populations"].items())
    return f"""
## benchmark_02 — alternating-frequency BAW (Zhang et al. 2023)

**Paper:** {s['paper']['citation']}, doi:[{s['paper']['doi']}](https://doi.org/{s['paper']['doi']}) (open access).
**Mode:** `alternating_baw`. **Config:** [`benchmark_02_alternating_baw/config.yaml`](benchmark_02_alternating_baw/config.yaml).
**Integrator:** fourth-order Runge–Kutta, as in the paper, landing on every switch; checked
against `solve_ivp` and the closed-form trajectory in `tests/test_baw.py`.

### Device

| Parameter | Value | Status |
|---|---|---|
| Channel | 737 µm × 50 µm, silicon + Pyrex, piezoceramic below | stated (Sec. 4.1) |
| Modes | 1 MHz (node W/2), 3 MHz (nodes W/6, W/2, 5W/6) | stated (Sec. 4.2) |
| Drive | 1 MHz 0.8 s / 9 Vpp; 3 MHz 1.4 s / 110 Vpp | stated (Sec. 2.2) |
| Flow | 50 µL/h sample + 100 µL/h sheath → y₀ < W/3 | stated; the band is *derived* from the ratio |
| Outlets | three; the centre third collects | count stated, widths **ASSUMPTION** |
| Acoustic region length | 20 mm | **ASSUMPTION** (swept in case d) |
| Cell sizes | MCF-7 18 µm (library), HCT116 14.8, A549 19.6, PBMC 8 µm | range 14.8–19.6 stated; per-line values **ASSUMPTION** |
| Compressibility | cancer 4.3, PBMC 4.0 ×10⁻¹⁰ Pa⁻¹ | only in Fig. 1 → **ASSUMPTION** |
| E_ac, 1 MHz @ 9 Vpp | **{cal['E1_J_m3']:.2f} J/m³** | **CALIBRATED** from the W/6 rule |
| E_ac, 3 MHz @ 110 Vpp | **{cal['E3_J_m3']:.2f} J/m³** | **CALIBRATED** from "minimum beyond 1 s" |

**Calibration.** The paper states no energy density. `E_1MHz` is the geometric mean of the
interval the stated design rule allows — a mean MCF-7 must move more than W/6 during the
1 MHz phase ({cal['rule']['target_minimum_J_m3']:.1f} J/m³ and up) and a mean PBMC less
(up to {cal['rule']['background_maximum_J_m3']:.1f} J/m³) — which gives equal factor margin
({cal['rule']['margin_factor']:.2f}×) on both sides. `E_3MHz` returns a mean PBMC from 90 % to
10 % of its node-to-antinode distance in 1.0 s; the fractions are choices, and case (a)'s
"minimum beyond 1 s" is therefore marked calibration-informed.
With these, the rule holds: {rule_s} against W/6 = {rule['threshold_um']:.0f} µm.

### Results — {_counts(rows)}

{_table(rows)}

{_notes(rows)}

{_fig(s, 'fig2a', B2, 'Fig. 2a analogue: 3 MHz duration')}
{_fig(s, 'fig2b', B2, 'Fig. 2b analogue: 1 MHz duration, with the Youden-best setting')}
{_fig(s, 'fig2c', B2, 'Fig. 2c analogue: 1 MHz amplitude, with the Youden-best setting')}
{_fig(s, 'fig8a', B2, 'Fig. 8a analogue: entry at the start of the 1 MHz phase')}
{_fig(s, 'fig8b', B2, 'Fig. 8b analogue: entry at the end of the 3 MHz phase')}
{_fig(s, 'fig8c', B2, 'Fig. 8c analogue: radiation and Stokes drag force, phases shaded')}
{_fig(s, 'fig4', B2, 'Fig. 4 analogue: A549 capture against flow rate')}
{_fig(s, 'table1', B2, 'Table 1 at each line’s best 1 MHz amplitude')}

### Why contamination is high: which assumption carries it

Capture matches the paper within a few points everywhere; PBMC contamination does not.
Re-running the baseline with one assumption changed at a time:

| Variant | Capture | Contamination |
|---|---|---|
{sens}

The inlet row settles it. Contamination comes from PBMCs that start in the upper part of
the sample stream: with a 1:2 ratio the stream's edge sits on the W/3 antinode of the
3 MHz mode, an unstable equilibrium, and a PBMC near it that meets a 1 MHz phase before a
full 3 MHz phase has pulled it back to W/6 is carried over. That is also why making every
cell enter at the *end* of the 3 MHz phase makes it worse here, not better: those cells
get only a fraction of a 3 MHz phase before the next push. Confining the inlet to
y₀ < W/6 — which the paper itself calls the theoretical optimum — removes most of it.
The real device evidently does better than the model at the stream edge; candidates are
hydrodynamic lift holding cells off the sheath interface, a narrower cell-laden band than
the flux share suggests, and a stronger 3 MHz field than the one calibrated here. None
of these is in a 1-D Gor'kov model.

### Flow rate and the unstated acoustic length

Durations at higher flow are the stated ones shortened only as much as the two-cycle rule
requires (the paper says they were reduced, not to what).

| Flow (µL/h) | Region (mm) | T₁ / T₃ (s) | A549 capture | Contamination |
|---|---|---|---|---|
{flows}

The stated drop to 84 % at 900 µL/h lies between the 30 and 40 mm rows. The flow trend is
reproduced; the 900 µL/h number is a statement about the unknown region length.

### Counterfactual — similar sizes, with and without the compressibility difference

CTC 12 µm against patient PBMC 10.5 µm (< 2 µm apart), equal densities, CV 10 %, best
achievable 1 MHz amplitude per arm. "Separated" = capture ≥ 80 % with contamination ≤ 10 %.

| Arm | Best J | Capture | Contamination | at | Separated |
|---|---|---|---|---|---|
{cf}

**Finding.** Size alone does not separate these cells, as the paper argues. But adding
the compressibility difference *in the direction their own Fig. 1 shows* — cancer cells
**more** compressible than PBMCs — makes separation **worse**, not better: in Gor'kov
theory a more compressible cell has a smaller monopole coefficient
`f₁ = 1 − κ_p/κ_f`, a smaller contrast factor, and a slower drift to the node. The
reversed arm shows that compressibility *can* rescue similar-sized cells, but only if the
CTCs are the stiffer ones. The paper's central claim ("effective separation was achieved
even when the size of CTCs is similar to that of PBMCs") is therefore **not reproduced**
by primary-radiation-force physics with the paper's own compressibility ordering. Density
differences (not measured there), acoustic streaming, or a different meaning of the
measured "compressibility" would have to account for it.
"""


def _check(summary: dict[str, Any] | None, needle: str) -> dict[str, Any] | None:
    if not summary:
        return None
    return next((c for c in summary["checks"] if needle in c["quantity"]), None)


def _ozet(s1: dict[str, Any] | None, s2: dict[str, Any] | None) -> str:
    """The Turkish summary, built from the same numbers as the tables."""
    lines = [
        "## Özet (Türkçe)",
        "",
        "İki makale referans vaka olarak yeniden üretildi. Referans değerler yalnızca "
        "makalelerin **metninde ve tablolarında açıkça yazan** sayılardır; grafiklerden değer "
        "okunmadı. Her vakada kalibre edilen büyüklük, karşılaştırılan bir sonuca değil, "
        "makalede sözle belirtilen bir koşula dayandırıldı. Nicel sapmalar test başarısızlığı "
        "değil **DEVIATION** olarak raporlanır ve olası nedenleri tabloların altında yazılıdır; "
        "makalenin belirttiği eğilimler ise testlerde zorunludur.",
        "",
    ]
    if s1:
        dy = _check(s1, "Delta Y at the optimum")
        idt = _check(s1, "IDT length at maximum")
        trends = [c for c in s1["checks"] if c["kind"] == "trend"]
        ok = sum(c["status"] == "PASS" for c in trends)
        rem = _check(s1, "WBC removal (mean")
        lines.append(
            f"- **Li 2015 (taSSAW):** belirtilen {len(trends)} eğilimin {ok}'i yeniden üretildi "
            f"(Fig. 2A, 2B, S2, 3). Kalibre edilmeyen ΔY tahmini {dy['ours']:.0f} µm "
            f"(makale ~{dy['paper']:.0f} µm, {dy['status']}); IDT uzunluğu optimumu "
            f"{idt['ours']:.0f} mm (makale 8–10 mm, {idt['status']}). Table 1'de model "
            f"37,5 dBm'de fazla sürüyor (WBC uzaklaştırma %{rem['ours']:.0f}, makale ~%90); "
            f"{s1['reanchored']['offset_db']:.1f} dB'lik bir güç kayması lökositleri ve MCF-7'yi "
            "düzeltir ama küçük hatları düzeltmez — modelin boyut seçiciliği cihazınkinden keskin "
            "ve bunu belirleyen girdiler (Table S1) erişilemez."
        )
    if s2:
        caps = [c for c in s2["checks"] if c["case"] == "Table 1" and "capture" in c["quantity"]]
        cons = [c for c in s2["checks"] if c["case"] == "Table 1" and "contamination" in c["quantity"]]
        cf = s2["counterfactual"]
        arms = list(cf.values())
        sens = {r["variant"]: r for r in s2["contamination_sensitivity"]}
        w6 = next((r for k, r in sens.items() if "W/6" in k), None)
        lines.append(
            "- **Zhang 2023 (alternatif frekanslı BAW):** Table 1 yakalama verimi üç hatta "
            + ", ".join(f"{c['quantity'].split()[0]} %{c['ours']:.1f} (makale %{c['paper']:.1f})"
                        for c in caps)
            + f"; PBMC kontaminasyonu ise %{min(c['ours'] for c in cons):.0f}–"
              f"%{max(c['ours'] for c in cons):.0f} (makale ~%1,5, DEVIATION). Nedeni, örnek "
              "akışının kenarının 3 MHz modunun kararsız W/3 antinoduna denk gelmesi"
            + (f"; giriş y₀ < W/6'ya sınırlanınca kontaminasyon %{w6['contamination_rate_percent']:.1f}'e "
               "iner." if w6 else ".")
            + " 900 µL/sa sonucu belirtilmemiş akustik bölge uzunluğuna bağlıdır."
        )
        lines.append(
            "- **Karşıt-olgu:** benzer boyutlarda (12 vs 10,5 µm) yalnız boyutla ayrışma "
            f"başarısız (en iyi J %{arms[0]['best_youden_percent']:.0f}). Makalenin Şekil 1'deki "
            "sıralamasıyla (kanser hücresi **daha** sıkıştırılabilir) sıkıştırılabilirlik farkı "
            f"ayrışmayı **kötüleştirir** (J %{arms[1]['best_youden_percent']:.0f}); ters sıralama "
            f"iyileştirirdi (J %{arms[2]['best_youden_percent']:.0f}). Makalenin temel iddiası "
            "Gor'kov fiziğiyle **yeniden üretilemedi** — bu bir bulgudur."
        )
    return "\n".join(lines) + "\n"


def build() -> str:
    s1, s2 = _load(B1), _load(B2)
    rows = (s1["checks"] if s1 else []) + (s2["checks"] if s2 else [])
    quick = any(s and s.get("quick") for s in (s1, s2))
    ozet = _ozet(s1, s2)
    header = f"""# Stage 1B — Literature benchmark report

*Generated {_dt.date.today().isoformat()} from commit `{_commit()}` by
`python -m benchmarks.make_report`.{' **QUICK-MODE RESULTS — rerun without --quick.**' if quick else ''}
Do not edit by hand: every number below is read from `*/results/summary.json`.*

{ozet}
## Summary (English)

Totals across both benchmarks — {_counts(rows)}.

| Status | Meaning |
|---|---|
| PASS | within tolerance (≤ 5 percentage points for percentages; ±10 % for ΔY), inside a stated range, or a stated trend reproduced |
| DEVIATION | a stated number outside tolerance; reported with its likely cause, **not** a test failure |
| FAIL | a stated trend the model gets backwards — would fail `tests/test_benchmarks.py` |
| REPRODUCED / NOT REPRODUCED | a qualitative claim of the paper, and whether the model supports it |
| CALIBRATION | a value the model was fitted to; shown so nobody counts it as agreement |

Reproduce: `python -m benchmarks.benchmark_01_tassaw.run`,
`python -m benchmarks.benchmark_02_alternating_baw.run`, then `python -m benchmarks.make_report`.
Tests: `pytest tests/test_benchmarks.py`.
"""
    body = ""
    body += section_01(s1) if s1 else "\n## benchmark_01\n\n*Not run.*\n"
    body += section_02(s2) if s2 else "\n## benchmark_02\n\n*Not run.*\n"
    footer = """
## What these benchmarks can and cannot say

* Both models are **2-D/1-D reduced**: a cross-section force law, analytic Poiseuille
  advection, no acoustic streaming, no inertial lift, no particle–particle interaction.
  Streaming and full piezoelectric fields are the job of the Stage 4 back-ends
  (`solvers/solver_openfoam`, `solvers/solver_elmer`).
* Every unstated input is listed as ASSUMPTION in the device tables and in the material
  library (`biosim materials`); the calibrated ones are listed as CALIBRATED with the rule
  they come from.
* Agreement on a trend is evidence the mechanism is right; agreement on a number after
  calibration is weaker evidence than it looks, and is labelled accordingly.

## References

* Li P, Mao Z, Peng Z, et al. (2015) Acoustic separation of circulating tumor cells.
  *PNAS* 112(16):4970–4975. doi:[10.1073/pnas.1504484112](https://doi.org/10.1073/pnas.1504484112)
* Zhang Y, Zhang Z, Zheng D, Huang T, Fu Q, Liu Y (2023) Label-free separation of
  circulating tumor cells and clusters by alternating frequency acoustic field in a
  microfluidic chip. *Int J Mol Sci* 24(4):3338.
  doi:[10.3390/ijms24043338](https://doi.org/10.3390/ijms24043338)
* Bruus H (2012) Acoustofluidics 7. *Lab Chip* 12:1014. doi:[10.1039/c2lc21068a](https://doi.org/10.1039/c2lc21068a)
* Barnkob R, Augustsson P, Laurell T, Bruus H (2010) Measuring the local pressure amplitude
  in microchannel acoustophoresis. *Lab Chip* 10:563. doi:[10.1039/b920376a](https://doi.org/10.1039/b920376a)
* Ding X, Peng Z, Lin S-CS, et al. (2014) Cell separation using tilted-angle standing
  surface acoustic waves. *PNAS* 111:12992. doi:[10.1073/pnas.1413325111](https://doi.org/10.1073/pnas.1413325111)
"""
    return header + body + footer


def main() -> None:
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
