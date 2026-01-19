# Makefile for RingKernel Technical Paper
#
# Usage:
#   make          - Build PDF
#   make clean    - Remove build artifacts
#   make watch    - Continuous rebuild on changes (requires latexmk)
#   make arxiv    - Create arxiv submission tarball
#   make wordcount - Count words in paper

# Main document
MAIN = main
LATEX = pdflatex
BIBTEX = bibtex
LATEXMK = latexmk

# Source files
SECTIONS = $(wildcard sections/*.tex)
FIGURES = $(wildcard figures/*.pdf)
BIB = references.bib

# Output
PDF = $(MAIN).pdf
ARXIV_TAR = ringkernel-arxiv.tar.gz

# Default target
.PHONY: all
all: $(PDF)

# Build PDF
$(PDF): $(MAIN).tex $(SECTIONS) $(BIB) $(FIGURES)
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

# Quick build (no bibliography)
.PHONY: quick
quick:
	$(LATEX) $(MAIN)

# Clean build artifacts
.PHONY: clean
clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out
	rm -f $(MAIN).pdf $(MAIN).toc $(MAIN).lof $(MAIN).lot
	rm -f $(MAIN).fdb_latexmk $(MAIN).fls $(MAIN).synctex.gz
	rm -f sections/*.aux
	rm -f $(ARXIV_TAR)

# Deep clean (including PDF)
.PHONY: distclean
distclean: clean
	rm -f $(PDF)

# Continuous rebuild on changes
.PHONY: watch
watch:
	$(LATEXMK) -pvc -pdf $(MAIN)

# Create arxiv submission tarball
.PHONY: arxiv
arxiv: $(PDF)
	@echo "Creating arxiv submission..."
	@mkdir -p arxiv-submission
	@cp $(MAIN).tex arxiv-submission/
	@cp $(MAIN).bbl arxiv-submission/
	@cp -r sections arxiv-submission/
	@cp -r figures arxiv-submission/
	@tar -czf $(ARXIV_TAR) -C arxiv-submission .
	@rm -rf arxiv-submission
	@echo "Created $(ARXIV_TAR)"

# Word count (approximate)
.PHONY: wordcount
wordcount:
	@echo "Word count (approximate):"
	@cat $(MAIN).tex $(SECTIONS) | \
		sed 's/\\[a-zA-Z]*{[^}]*}//g' | \
		sed 's/\\[a-zA-Z]*//g' | \
		sed 's/[{}]//g' | \
		wc -w

# Check for common LaTeX issues
.PHONY: check
check:
	@echo "Checking for common issues..."
	@grep -n "TODO" $(MAIN).tex $(SECTIONS) || true
	@grep -n "FIXME" $(MAIN).tex $(SECTIONS) || true
	@grep -n "XXX" $(MAIN).tex $(SECTIONS) || true
	@echo "Checking for undefined references..."
	@grep -c "undefined" $(MAIN).log || echo "No undefined references"

# Spell check (requires aspell)
.PHONY: spell
spell:
	@for f in $(SECTIONS); do \
		echo "Checking $$f..."; \
		aspell --mode=tex --lang=en check $$f; \
	done

# View PDF (Linux)
.PHONY: view
view: $(PDF)
	xdg-open $(PDF) 2>/dev/null || open $(PDF) 2>/dev/null || evince $(PDF)

# Help
.PHONY: help
help:
	@echo "RingKernel Paper Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build PDF (default)"
	@echo "  quick     - Quick build without bibliography"
	@echo "  clean     - Remove build artifacts"
	@echo "  distclean - Remove all generated files including PDF"
	@echo "  watch     - Continuous rebuild on changes"
	@echo "  arxiv     - Create arxiv submission tarball"
	@echo "  wordcount - Approximate word count"
	@echo "  check     - Check for TODOs and undefined references"
	@echo "  spell     - Run spell checker on sections"
	@echo "  view      - Open PDF in viewer"
	@echo "  help      - Show this help"
	@echo ""
	@echo "Requirements:"
	@echo "  - pdflatex, bibtex"
	@echo "  - latexmk (for watch target)"
	@echo "  - aspell (for spell target)"
