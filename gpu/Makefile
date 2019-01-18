all: compile run
test: compile-test run
compile:
	g++ -lOpenCL -O3 main.cpp -o main.out

run:
	./main.out 10640 3730 500 0.001 epoch-scalability/500 16 1024

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
	mkdir results/epoch-scalability results/size-scalability results/local-work-scalability results/global-work-scalability
	cd results/epoch-scalability &&	mkdir 1000 500 250 && cd ../..
	cd results/size-scalability && mkdir 15640 11730 5865 && cd ../..
	cd results/local-work-scalability && mkdir 16 8 4 2 1 && cd ../..
	cd results/global-work-scalability && mkdir 1024 512 256 128 64 && cd ../..


scalability-epoch-experiments:
	./main.out 10640 3730 1000 0.001 epoch-scalability/1000 16 1024; \
	./main.out 10640 3730 500 0.001 epoch-scalability/500 16 1024; \
	./main.out 10640 3730 250 0.001 epoch-scalability/250 16 1024; \

scalability-size-experiments:
	./main.out 10640 3730 500 0.001 size-scalability/15640 16 1024; \
	./main.out 5640 3730 500 0.001 size-scalability/11730 16 1024; \
	./main.out 2640 3730 500 0.001 size-scalability/5865 16 1024; \

scalability-local-experiments:
	./main.out 10640 3730 500 0.001 local-work-scalability/16 16 1024; \
	./main.out 10640 3730 500 0.001 local-work-scalability/8 8 1024; \
	./main.out 10640 3730 500 0.001 local-work-scalability/4 4 1024; \
	./main.out 10640 3730 500 0.001 local-work-scalability/2 2 1024; \
	./main.out 10640 3730 500 0.001 local-work-scalability/1 1 1024; \


scalability-global-experiments:
	./main.out 10640 3730 500 0.001 global-work-scalability/1024 16 1024; \
	./main.out 10640 3730 500 0.001 global-work-scalability/512 16 512; \
	./main.out 10640 3730 500 0.001 global-work-scalability/256 16 256; \
	./main.out 10640 3730 500 0.001 global-work-scalability/128 16 128; \
	./main.out 10640 3730 500 0.001 global-work-scalability/64 16 64; \

	
experiments: compile scalability-size-experiments scalability-local-experiments scalability-global-experiments

documentation:
	doxygen Doxyfile
