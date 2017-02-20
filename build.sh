#!/bin/bash

# use `FMT=tex ./build-paper.sh` to get the .tex file

export TEXINPUTS=wissdoc:

rsync -a --delete img bib.bib net1.tex wissdoc/ build

pandoc \
	thesis.md \
	--natbib \
	--filter pandoc-crossref \
	--standalone \
	--template wissdoc/diplarb.tex \
	--top-level-division=chapter \
	-o build/thesis.tex

(
	cd build
	texfot pdflatex -interaction=batchmode thesis.tex
	bibtex thesis
	texfot pdflatex -interaction=batchmode thesis.tex
	echo -e "\e[31mFinal run\e[0m"
	texfot pdflatex -interaction=nonstopmode thesis.tex
)
