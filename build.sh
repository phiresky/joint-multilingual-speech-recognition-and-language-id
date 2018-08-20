#!/bin/bash

export TEXINPUTS=wissdoc:

rsync -a --delete img tbl template/ bib.bib build

pandoc \
	paper.md \
	--filter pandoc-citeproc \
	--bibliography=bib.bib \
	--filter pandoc-crossref \
	--standalone \
	-o build/paper.pdf

: '
(
	cd build
	texfot pdflatex -interaction=batchmode thesis.tex >/dev/null
	bibtex thesis
	texfot pdflatex -interaction=batchmode thesis.tex >/dev/null
	texfot pdflatex -interaction=batchmode thesis.tex >/dev/null
	echo -e "\e[31mFinal run\e[0m"
	texfot pdflatex -interaction=nonstopmode thesis.tex
)

'