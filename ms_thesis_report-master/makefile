

rebuild: clean report

report:
	pdflatex -output-directory=build 0_thesis
ifeq ($(OS),Windows_NT)
	echo "here"
	copy 7_references.bib build
else
	cp 7_references.bib build
endif	
	cd .\build &&  bibtex 0_thesis

ifneq ($(OS),Windows_NT)
	cd ../
endif
	pdflatex -output-directory=build 0_thesis
	pdflatex -output-directory=build 0_thesis
ifeq ($(OS),Windows_NT)
	del thesis.pdf
	copy build\0_thesis.pdf thesis.pdf
else
	mv build/0_thesis.pdf thesis.pdf
	open thesis.pdf
endif
	echo Done.	

clean:
ifeq ($(OS),Windows_NT)
	del /Q thesis.pdf
	del /Q build\* 
else
	rm -f thesis.pdf
	rm -rf build
	mkdir build
endif