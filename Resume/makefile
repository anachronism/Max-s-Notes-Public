

resume:
	pdflatex -output-directory=build mhli_resume
	cd build 
	pdflatex -output-directory=build mhli_resume
	pdflatex -output-directory=build mhli_resume
ifeq ($(OS),Windows_NT)
	del mhli_resume.pdf
	copy build\mhli_resume.pdf mhli_resume.pdf
else
	mv build/mhli_resume.pdf mhli_resume.pdf
	open mhli_resume.pdf
endif
	echo Done.	


rebuild: clean report

clean:
ifeq ($(OS),Windows_NT)
	del mhli_resume.pdf
	del build\*
else
	rm -f mhli_resume.pdf
	rm -rf build
	mkdir build
endif