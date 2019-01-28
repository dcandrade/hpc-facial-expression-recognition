all: compile run
test: compile-test run
compile:
	g++ -lOpenCL -O3 main.cpp -o main.out

run:
	./main.out 10900 3730 500 0.001 epoch-scalability/500 256 1024

debug:
	g++ -g main.cpp -o main.out
	gdb main.out

compile-test:
	g++ -lOpenCL -O3 test_gpu.cpp -o main.out

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
	mkdir results/size-scalability results/local-work-scalability
	cd results/size-scalability && mkdir 10900 8175 5450 && cd ../..
	cd results/local-work-scalability && mkdir 256 128 64 32 16 && cd ../..

scalability-size-experiments:
	./main.out 10900 3730 500 0.001 size-scalability/10900 256; \
	./main.out 8175 3730 500 0.001 size-scalability/8175 256; \
	./main.out 5450 3730 500 0.001 size-scalability/5450 256; \

scalability-local-experiments:
	./main.out 10900 3730 500 0.001 local-work-scalability/256 256; \
	./main.out 10900 3730 500 0.001 local-work-scalability/128 128; \
	./main.out 10900 3730 500 0.001 local-work-scalability/64 64; \
	./main.out 10900 3730 500 0.001 local-work-scalability/32 32; \
	./main.out 10900 3730 500 0.001 local-work-scalability/16 16; \
	
experiments: compile scalability-size-experiments scalability-local-experiments

documentation:
	doxygen Doxyfile