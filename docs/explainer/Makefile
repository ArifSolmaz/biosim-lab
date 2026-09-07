# Build the explainer PDFs, English and Turkish.
#   make          -> both
#   make en       -> biosim-lab-explained.pdf
#   make tr       -> biosim-lab-nasil-calisir.pdf
#   make clean    -> remove build artefacts
#
# Needs xelatex (TeX Live / MacTeX) and the figures in ../../assets/, which the
# examples produce:
#   python examples/02_ctc_vs_rbc.py
#   python examples/04_validate_analytic_vs_fem.py
#   python examples/05_impedance_rtca.py

EN = biosim-lab-explained.pdf
TR = biosim-lab-nasil-calisir.pdf

all: en tr

en: $(EN)
tr: $(TR)

# Twice, so the table of contents resolves.
$(EN): main.tex
	xelatex -interaction=nonstopmode main.tex > /dev/null
	xelatex -interaction=nonstopmode main.tex > /dev/null
	mv main.pdf $(EN)

$(TR): main-tr.tex
	xelatex -interaction=nonstopmode main-tr.tex > /dev/null
	xelatex -interaction=nonstopmode main-tr.tex > /dev/null
	mv main-tr.pdf $(TR)

clean:
	rm -f main.aux main.log main.out main.toc main.pdf
	rm -f main-tr.aux main-tr.log main-tr.out main-tr.toc main-tr.pdf

.PHONY: all en tr clean
