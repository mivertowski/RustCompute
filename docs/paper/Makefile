# Makefile — Persistent GPU Actors paper
#
# Usage:
#   make            # build PDF
#   make quick      # one-pass build (no bibliography refresh)
#   make clean      # remove auxiliary files (keep PDF)
#   make distclean  # remove everything (including PDF)
#   make watch      # rebuild on every save (requires latexmk)

PAPER     = main
LATEX     = pdflatex
BIBTEX    = bibtex
LATEXMK   = latexmk

LATEX_OPTS = -interaction=nonstopmode -halt-on-error -file-line-error

SECTIONS = $(wildcard sections/*.tex)
FIGURES  = $(wildcard figures/*)
BIB      = references.bib

.PHONY: all quick clean distclean watch verify

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex $(SECTIONS) $(BIB) $(FIGURES)
	$(LATEX) $(LATEX_OPTS) $(PAPER)
	$(BIBTEX) $(PAPER)
	$(LATEX) $(LATEX_OPTS) $(PAPER)
	$(LATEX) $(LATEX_OPTS) $(PAPER)

quick: $(PAPER).tex $(SECTIONS)
	$(LATEX) $(LATEX_OPTS) $(PAPER)

watch:
	$(LATEXMK) -pdf -pvc -interaction=nonstopmode $(PAPER).tex

# Verify the TLA+ specs the paper depends on (run from repo root, not paper/).
verify:
	cd ../verification && ./tlc.sh

clean:
	rm -f $(PAPER).aux $(PAPER).log $(PAPER).out $(PAPER).toc
	rm -f $(PAPER).bbl $(PAPER).blg $(PAPER).fls $(PAPER).fdb_latexmk
	rm -f $(PAPER).synctex.gz
	rm -f sections/*.aux

distclean: clean
	rm -f $(PAPER).pdf
