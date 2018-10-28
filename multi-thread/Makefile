all: compile run
compile:
	g++ -fopenmp -O3 main.cpp -o main.out

run:
	./main.out 15640 3730 200 0.001 epoch-scalability/200

debug:
	g++ -g main.cpp -o main.out
	gdb main.out

clean:
	rm *.out
	rm -rf results
	rm perf.*

profile: compile
	perf record -e cycles,branches,branch-misses,cache-misses,bus-cycles,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,dTLB-loads,dTLB-load-misses -d ./main.out 15640 3730 200 0.001 epoch-scalability/100

setup-experiments:
	mkdir results/
	mkdir results/epoch-scalability results/size-scalability
	cd results/epoch-scalability &&	mkdir 800 400 200 100 50 && cd ../..
	cd results/size-scalability && mkdir 15640 7820 3910 1955 977 && cd ../..

scalability-epoch-experiments:
	for i in {1..5}; do \
		./main.out 15640 3730 400 0.001 epoch-scalability/400; \
		./main.out 15640 3730 200 0.001 epoch-scalability/200; \
		./main.out 15640 3730 100 0.001 epoch-scalability/100; \
		./main.out 15640 3730 50 0.001 epoch-scalability/50; \
		./main.out 15640 3730 25 0.001 epoch-scalability/800; \
	done;

scalability-size-experiments:
	for i in {1..5}; do \
		./main.out 15640 3730 200 0.001 size-scalability/15640; \
		./main.out 7820 3730 200 0.001 size-scalability/7820; \
		./main.out 3910 3730 200 0.001 size-scalability/3910; \
		./main.out 1955 3730 200 0.001 size-scalability/1955; \
		./main.out 977 3730 200 0.001 size-scalability/977; \
	done;
	
experiments: compile scalability-epoch-experiments scalability-size-experiments

documentation:
	doxygen Doxyfile
