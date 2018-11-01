all: compile run
compile:
	g++ -fopenmp -O3 main.cpp -o main.out

run:
	./main.out 15640 3730 500 0.001 epoch-scalability/500 4

debug:
	g++ -g main.cpp -o main.out
	gdb main.out

clean:
	rm *.out
	rm -rf results
	rm perf.*

profile: 
	g++  -g -pg -fopenmp -O3 -o main.out main.cpp
	./main.out
	gprof main.out > profile.txt

setup-experiments:
	mkdir results/
	mkdir results/epoch-scalability results/size-scalability
	cd results/epoch-scalability &&	mkdir 100 200 50 25 && cd ../..
	cd results/size-scalability && mkdir 15640 11730 5865 && cd ../..
	cd results/thread-scalability && mkdir 1 2 3 4 && cd ../..
scalability-epoch-experiments:
	for i in {1..5}; do \
		./main.out 15640 3730 1000 0.001 epoch-scalability/100 4; \
		./main.out 15640 3730 500 0.001 epoch-scalability/50 4; \
		./main.out 15640 3730 250 0.001 epoch-scalability/25 4; \
		./main.out 15640 3730 125 0.001 epoch-scalability/200 4; \
		#./main.out 15640 3730 50 0.001 epoch-scalability/50 4; \
	done;

scalability-size-experiments:
	for i in {1..5}; do \
		./main.out 15640 3730 100 0.001 epoch-scalability/15640 4; \
		./main.out 11730 3730 100 0.001 epoch-scalability/11730 4; \
		./main.out 7280 3730 100 0.001 epoch-scalability/5865 4; \
	done;
	
scalability-thread-experiments:
	for i in {1..5}; do \
		./main.out 15640 3730 100 0.001 thread-scalability/1 1; \
		./main.out 7820 3730 100 0.001 thread-scalability/2 2; \
		./main.out 3910 3730 100 0.001 thread-scalability/3 3; \
		./main.out 1955 3730 100 0.001 thread-scalability/4 4; \
	done;

experiments: compile scalability-epoch-experiments scalability-size-experiments scalability-thread-experiments

documentation:
	doxygen Doxyfile
