all: compile run

compile:
	g++ -O3 main.cpp -o main.out

run:
	./main.out

debug:
	g++ -g main.cpp -o main.out
	gdb main.out

clean:
	rm *.out
	rm -rf results
	rm perf.*

profile: compile
	perf record -e cycles,instructions,cache-references,branches,branch-misses,cache-misses,bus-cycles,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,dTLB-loads,dTLB-load-misses -d ./main.out

setup-experiments:
	mkdir results/
	mkdir results/epoch-scalability results/size-scalability
	cd results/epoch-scalability &&	mkdir 400 200 100 50 25 && cd ../..
	cd results/size-scalability && mkdir 8k 4k 2k 1k 500 && cd ../..

scalability-epoch-experiments:
	for i in {1..5}; do \
		(time ./main.out 8092 2012 400 0.01 epoch-scalability/400) 2>> results/epoch-scalability/400/time400.txt; \
		(time ./main.out 8092 2012 200 0.01 epoch-scalability/200) 2>> results/epoch-scalability/200/time200.txt; \
		(time ./main.out 8092 2012 100 0.01 epoch-scalability/100) 2>> results/epoch-scalability/100/time100.txt; \
		(time ./main.out 8092 2012 50 0.01 epoch-scalability/50) 2>> results/epoch-scalability/50/time50.txt; \
		(time ./main.out 8092 2012 25 0.01 epoch-scalability/25) 2>> results/epoch-scalability/25/time25.txt; \
	done;

scalability-size-experiments:
	for i in {1..5}; do \
		(time ./main.out 8092 2012 100 0.01 size-scalability/8k) 2>> results/size-scalability/8k/time8k.txt; \
		(time ./main.out 4092 2012 100 0.01 size-scalability/4k) 2>> results/size-scalability/4k/time4k.txt; \
		(time ./main.out 2092 2012 100 0.01 size-scalability/2k) 2>> results/size-scalability/2k/time2k.txt; \
		(time ./main.out 1092 2012 100 0.01 size-scalability/1k) 2>> results/size-scalability/1k/time.txt; \
		(time ./main.out 592 2012 100 0.01 size-scalability/500) 2>> results/size-scalability/500/time.txt; \
	done;
	
experiments: compile scalability-epoch-experiments scalability-size-experiments
