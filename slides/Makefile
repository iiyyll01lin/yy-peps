# Marp build for the PEPS-on-AMD slide decks / PEPS-on-AMD 投影片建置
#
# Requires Node.js >=18 + npx. The exact Marp CLI version is pinned here; npx
# fetches it on first use, so there is nothing to install globally.
#
#   make                       # build every deck to PDF (same as `make all`)
#   make pdf                   # same as above
#   make 03_applications.pdf   # build a single deck
#   make pptx                  # build every deck to PPTX
#   make html                  # build every deck to HTML
#   make validate              # CI build to slides/build/*.html
#   make version               # print the pinned marp-cli version
#   make clean                 # remove generated PDF/PPTX/HTML

MARP_VERSION := 4.5.0
MARP       ?= npx --yes @marp-team/marp-cli@$(MARP_VERSION)
MARP_FLAGS ?= --allow-local-files --html
VALIDATE_DIR ?= build

DECKS := $(basename $(notdir $(wildcard [0-9][0-9]_*.md)))
PDFS  := $(addsuffix .pdf,$(DECKS))
PPTXS := $(addsuffix .pptx,$(DECKS))
HTMLS := $(addsuffix .html,$(DECKS))
VALIDATED_HTMLS := $(addprefix $(VALIDATE_DIR)/,$(HTMLS))

.PHONY: all pdf pptx html validate version clean
.DEFAULT_GOAL := all

all: pdf
pdf:  $(PDFS)
pptx: $(PPTXS)
html: $(HTMLS)
validate: $(VALIDATED_HTMLS)

%.pdf: %.md
	$(MARP) $(MARP_FLAGS) $< -o $@

%.pptx: %.md
	$(MARP) $(MARP_FLAGS) $< -o $@

%.html: %.md
	$(MARP) $(MARP_FLAGS) $< -o $@

$(VALIDATE_DIR)/%.html: %.md
	mkdir -p $(VALIDATE_DIR)
	$(MARP) $(MARP_FLAGS) $< -o $@

version:
	$(MARP) --version

clean:
	rm -f $(PDFS) $(PPTXS) $(HTMLS)
	rm -rf $(VALIDATE_DIR)
