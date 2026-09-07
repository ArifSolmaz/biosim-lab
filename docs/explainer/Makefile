# Build the explainer PDF.
#   make          -> biosim-lab-explained.pdf
#   make clean    -> remove build artefacts
#
# Needs xelatex (TeX Live / MacTeX) and the figures in ../../assets/, which are
# produced by running the examples:
#   python examples/02_ctc_vs_rbc.py
#   python examples/04_validate_analytic_vs_fem.py
#   python examples/05_impedance_rtca.py

PDF = biosim-lab-explained.pdf

$(PDF): main.tex
	xelatex -interaction=nonstopmode main.tex > /dev/null
	xelatex -interaction=nonstopmode main.tex > /dev/null
	mv main.pdf $(PDF)

clean:
	rm -f main.aux main.log main.out main.toc main.pdf

.PHONY: clean
