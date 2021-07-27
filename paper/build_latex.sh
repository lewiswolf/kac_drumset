cd paper
	# pdflatex -> bibtex -> pdflatex -> pdflatex
	pipenv run pdflatex --shell-escape -synctex=1 -interaction=nonstopmode -file-line-error index.tex
	pipenv run bibtex bib.bib
	pipenv run pdflatex --shell-escape -synctex=1 -interaction=nonstopmode -file-line-error index.tex
	pipenv run pdflatex --shell-escape -synctex=1 -interaction=nonstopmode -file-line-error index.tex
	
	# clean up auxillary files
	rm index.aux
	rm document.aux
	rm index.log
	rm index.out