all: compile run
compile:
	g++ -fopenmp -O3 main.cpp -o main.out

run:
	./main.out 15640 3730 200 0.001 epoch-scalability/200 4

debug:
	g++ -g main.cpp -o main.out
	gdb main.out

clean:
	rm *.out
	rm -rf results
	rm perf.*

profile: compile
	perf record -e cycles,branches,branch-misses,cache-misses,bus-cycles,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,dTLB-loads,dTLB-load-misses -d ./main.out 15640 3730 200 0.001 epoch-scalability/200 4

setup-experiments:
	mkdir results/
	mkdir results/epoch-scalability results/size-scalability
	cd results/epoch-scalability &&	mkdir 800 400 200 100 50 && cd ../..
	cd results/size-scalability && mkdir 15640 7820 3910 1955 977 && cd ../..
	cd results/thread-scalability && mkdir 1 2 3 4 && cd ../..


scalability-epoch-experiments:
	for i in {1..5}; do \
		./main.out 15640 3730 400 0.001 epoch-scalability/400 4; \
		./main.out 15640 3730 200 0.001 epoch-scalability/200 4; \
		./main.out 15640 3730 100 0.001 epoch-scalability/100 4; \
		./main.out 15640 3730 50 0.001 epoch-scalability/50 4; \
		./main.out 15640 3730 25 0.001 epoch-scalability/800 4; \
	done;

scalability-size-experiments:
	for i in {1..5}; do \
		./main.out 15640 3730 200 0.001 size-scalability/15640 4; \
		./main.out 7820 3730 200 0.001 size-scalability/7820 4; \
		./main.out 3910 3730 200 0.001 size-scalability/3910 4; \
		./main.out 1955 3730 200 0.001 size-scalability/1955 4; \
		./main.out 977 3730 200 0.001 size-scalability/977 4; \
	done;
	
scalability-thread-experiments:
	for i in {1..5}; do \
		./main.out 15640 3730 200 0.001 thread-scalability/1 1; \
		./main.out 7820 3730 200 0.001 thread-scalability/2 2; \
		./main.out 3910 3730 200 0.001 thread-scalability/3 3; \
		./main.out 1955 3730 200 0.001 thread-scalability/4 4; \
	done;

experiments: compile scalability-epoch-experiments scalability-size-experiments scalability-thread-experiments

documentation:
	doxygen Doxyfile
