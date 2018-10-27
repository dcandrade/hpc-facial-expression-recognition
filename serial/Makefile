all: compile run
compile:
	g++ -O3 main.cpp -o main.out

run:
	./main.out 8092 2012 100 0.01 epoch-scalability/100

debug:
	g++ -g main.cpp -o main.out
	gdb main.out

clean:
	rm *.out
	rm -rf results
	rm perf.*

profile: compile
	perf record -e cycles,branches,branch-misses,cache-misses,bus-cycles,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,dTLB-loads,dTLB-load-misses -d ./main.out 8092 2012 100 0.01 epoch-scalability/100

setup-experiments:
	mkdir results/
	mkdir results/epoch-scalability results/size-scalability
	cd results/epoch-scalability &&	mkdir 400 200 100 50 25 && cd ../..
	cd results/size-scalability && mkdir 8k 4k 2k 1k 500 && cd ../..

scalability-epoch-experiments:
	for i in {1..5}; do \
		./main.out 8092 2012 400 0.01 epoch-scalability/400; \
		./main.out 8092 2012 200 0.01 epoch-scalability/200; \
		./main.out 8092 2012 100 0.01 epoch-scalability/100; \
		./main.out 8092 2012 50 0.01 epoch-scalability/50; \
		./main.out 8092 2012 25 0.01 epoch-scalability/25; \
	done;

scalability-size-experiments:
	for i in {1..5}; do \
		./main.out 8092 2012 100 0.01 size-scalability/8k; \
		./main.out 4092 2012 100 0.01 size-scalability/4k; \
		./main.out 2092 2012 100 0.01 size-scalability/2k; \
		./main.out 1092 2012 100 0.01 size-scalability/1k; \
		./main.out 592 2012 100 0.01 size-scalability/500; \
	done;
	
experiments: compile scalability-epoch-experiments scalability-size-experiments

documentation:
	doxygen Doxyfile
