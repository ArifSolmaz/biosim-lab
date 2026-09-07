# The explainer / Anlatım

Two editions of the same document:

| File | Language | Pages |
|---|---|---|
| `biosim-lab-explained.pdf` (`main.tex`) | English | 29 |
| `biosim-lab-nasil-calisir.pdf` (`main-tr.tex`) | Türkçe | 31 |

A walk through the biology, physics, mathematics and computation behind this
project, written for someone who is not an expert in any of them.

Roughly a fifth of it is about what the model gets *wrong*, because that is the
part a reader needs in order to judge the rest.

## Contents

| Part | What it covers |
|---|---|
| I | The problem: finding one tumour cell among five billion blood cells; what a cell is to a physicist |
| II | Standing waves, why they push things, the contrast factor, the $r^2$ law that makes the device work, surface acoustic waves, and the two places the textbook formula gets it wrong here |
| III | The impedance instrument: cells as tiny insulators, Cell Index, IC50 |
| IV | Image analysis: watershed segmentation, Poisson counting error, mean-squared displacement |
| V | How a computer solves any of it: finite elements, ODE integration, and how the code is validated |
| VI | Every number in the default run, and an honest list of what is assumed or missing |
| Appendices | Symbol table, a units primer, and where each idea came from |

## Building it

```bash
make            # both editions; needs xelatex
make en         # English only
make tr         # Türkçe only
```

The Turkish edition uses `polyglossia` with `\setmainlanguage{turkish}` for
correct hyphenation. Both are built with XeLaTeX, which is what makes the
Turkish characters and the STIX Two maths font work together.

The figures come from `../../assets/`, which the examples generate:

```bash
python examples/02_ctc_vs_rbc.py
python examples/04_validate_analytic_vs_fem.py
python examples/05_impedance_rtca.py
```
