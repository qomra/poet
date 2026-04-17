# ============================================================================
# الشاعر / The Poet — LaTeX Report
# Requires: xelatex, latexmk, biber
# ============================================================================

REPORT   := report
BUILDDIR := build
PDF      := $(BUILDDIR)/$(REPORT).pdf

.DEFAULT_GOAL := pdf

# ── Build ────────────────────────────────────────────────────────────────────
.PHONY: pdf
pdf: $(BUILDDIR) ## Compile the full report to build/report.pdf
	latexmk \
		-lualatex \
		-bibtex \
		-outdir=$(BUILDDIR) \
		-interaction=nonstopmode \
		-halt-on-error \
		-f \
		$(REPORT).tex
	@echo ""
	@echo "✓  PDF ready: $(PDF)"

# ── Watch mode ───────────────────────────────────────────────────────────────
.PHONY: watch
watch: $(BUILDDIR) ## Recompile automatically on file changes
	latexmk \
		-lualatex \
		-bibtex \
		-outdir=$(BUILDDIR) \
		-interaction=nonstopmode \
		-pvc \
		$(REPORT).tex

# ── Quick compile (single pass, no bib) ──────────────────────────────────────
.PHONY: quick
quick: $(BUILDDIR) ## Single xelatex pass — fast, skips bibliography
	lualatex \
		-output-directory=$(BUILDDIR) \
		-interaction=nonstopmode \
		$(REPORT).tex

# ── Open PDF ─────────────────────────────────────────────────────────────────
.PHONY: open
open: ## Open the compiled PDF
	@test -f $(PDF) || (echo "Run 'make pdf' first"; exit 1)
	xdg-open $(PDF) 2>/dev/null || powershell.exe -c "Start-Process '$(shell wslpath -w $(PDF))'" 2>/dev/null || open $(PDF)

# ── Clean ────────────────────────────────────────────────────────────────────
.PHONY: clean
clean: ## Remove all build artifacts (keep PDF)
	latexmk -outdir=$(BUILDDIR) -c $(REPORT).tex 2>/dev/null || true
	rm -f $(BUILDDIR)/*.aux $(BUILDDIR)/*.log $(BUILDDIR)/*.toc \
	      $(BUILDDIR)/*.lof $(BUILDDIR)/*.lot $(BUILDDIR)/*.bbl \
	      $(BUILDDIR)/*.bcf $(BUILDDIR)/*.blg $(BUILDDIR)/*.run.xml \
	      $(BUILDDIR)/*.fdb_latexmk $(BUILDDIR)/*.fls $(BUILDDIR)/*.out \
	      $(BUILDDIR)/*.xdv $(BUILDDIR)/*.synctex.gz
	@echo "✓  Build artifacts removed (PDF preserved)"

.PHONY: cleanall
cleanall: ## Remove build directory entirely including PDF
	rm -rf $(BUILDDIR)
	@echo "✓  Build directory removed"

# ── Helpers ──────────────────────────────────────────────────────────────────
$(BUILDDIR):
	mkdir -p $(BUILDDIR)/chapters $(BUILDDIR)/appendices $(BUILDDIR)/figures

.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} \
	      /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
