#!/bin/bash

export TEXINPUTS=wissdoc:

rsync -a --delete img tbl template/ bib.bib build

pandoc \
	thesis.md \
	--natbib \
	--filter pandoc-crossref \
	--standalone \
	--template template/diplarb.tex \
	--top-level-division=chapter \
	--highlight-style monochrome \
	-o build/thesis.tex

(
	cd build
	texfot pdflatex -interaction=batchmode thesis.tex >/dev/null
	bibtex thesis
	texfot pdflatex -interaction=batchmode thesis.tex >/dev/null
	texfot pdflatex -interaction=batchmode thesis.tex >/dev/null
	echo -e "\e[31mFinal run\e[0m"
	texfot pdflatex -interaction=nonstopmode thesis.tex
)
