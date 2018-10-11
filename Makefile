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
	perf record -e cycles,instructions,cache-references,cache-misses,bus-cycles,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,dTLB-loads,dTLB-load-misses -d ./main.out

setup-experiments:
	mkdir results/
	mkdir results/epoch-scalability results/size-scalability

scalability-epoch-experiments:
	for i in {1..5}; do \
		(time ./main.out 8092 2012 400 0.01 epoch-scalability) 2>> results/epoch-scalability/time400.txt; \
		(time ./main.out 8092 2012 200 0.01 epoch-scalability) 2>> results/epoch-scalability/time200.txt; \
		(time ./main.out 8092 2012 100 0.01 epoch-scalability) 2>> results/epoch-scalability/time100.txt; \
		(time ./main.out 8092 2012 50 0.01 epoch-scalability) 2>> results/epoch-scalability/time50.txt; \
		(time ./main.out 8092 2012 25 0.01 epoch-scalability) 2>> results/epoch-scalability/time25.txt; \
	done;

scalability-size-experiments:
	for i in {1..5}; do \
		(time ./main.out 8092 2012 100 0.01 size-scalability) 2>> results/size-scalability/train8k.txt; \
		(time ./main.out 4092 2012 100 0.01 size-scalability) 2>> results/size-scalability/train4k.txt; \
		(time ./main.out 2092 2012 100 0.01 size-scalability) 2>> results/size-scalability/train2k.txt; \
		(time ./main.out 1092 2012 100 0.01 size-scalability) 2>> results/size-scalability/train1k.txt; \
		(time ./main.out 592 2012 100 0.01 size-scalability) 2>> results/size-scalability/train500.txt; \
	done;
	
experiments: compile scalability-epoch-experiments scalability-size-experiments
