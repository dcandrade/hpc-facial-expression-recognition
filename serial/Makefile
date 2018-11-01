all: compile run
compile:
	g++ -O3 main.cpp -o main.out

run:
	./main.out 15640 3730 100 0.001 epoch-scalability/100

debug:
	g++ -g main.cpp -o main.out
	gdb main.out

clean:
	rm *.out
	rm -rf results
	rm perf.*

profile: 
	g++  -g -pg -O3 -o main.out main.cpp
	./main.out
	gprof main.out > profile.txt

setup-experiments:
	mkdir results/
	mkdir results/epoch-scalability results/size-scalability
	cd results/epoch-scalability &&	mkdir 100 200 50 25 && cd ../..
	cd results/size-scalability && mkdir 15640 11730 5865 && cd ../..

scalability-epoch-experiments:
	./main.out 15640 3730 100 0.001 epoch-scalability/100; \
	./main.out 15640 3730 200 0.001 epoch-scalability/200; \
	./main.out 15640 3730 50 0.001 epoch-scalability/50; \
	./main.out 15640 3730 25 0.001 epoch-scalability/25; \

scalability-size-experiments:
	./main.out 15640 3730 100 0.001 epoch-scalability/15640; \
	./main.out 11730 3730 100 0.001 epoch-scalability/11730; \
	./main.out 7280 3730 100 0.001 epoch-scalability/5865; \

	
experiments: compile scalability-epoch-experiments scalability-size-experiments

documentation:
	doxygen Doxyfile
