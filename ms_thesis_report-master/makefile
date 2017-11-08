report:
	pdflatex -output-directory=build 0_thesis
	cp 7_references.bib build
	cd build && bibtex 0_thesis
	pdflatex -output-directory=build 0_thesis
	pdflatex -output-directory=build 0_thesis
	mv build/0_thesis.pdf thesis.pdf
	open thesis.pdf

rebuild: clean report

clean:
	rm -f thesis.pdf
	rm -rf build
	mkdir build
