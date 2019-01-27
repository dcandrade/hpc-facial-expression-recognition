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
	cd results/local-work-scalability && mkdir 256 128 64 32 16 && cd ../..
	cd results/global-work-scalability && mkdir 1024 512 256 128 64 && cd ../..


scalability-epoch-experiments:
	./main.out 10640 3730 1000 0.001 epoch-scalability/1000 256 1024; \
	./main.out 10640 3730 500 0.001 epoch-scalability/500 256 1024; \
	./main.out 10640 3730 250 0.001 epoch-scalability/250 12566 1024; \

scalability-size-experiments:
	./main.out 10640 3730 500 0.001 size-scalability/15640 256 1024; \
	./main.out 5640 3730 500 0.001 size-scalability/11730 256 1024; \
	./main.out 2640 3730 500 0.001 size-scalability/5865 256 1024; \

scalability-local-experiments:
	./main.out 10640 3730 500 0.001 local-work-scalability/256 256 1024; \
	./main.out 10640 3730 500 0.001 local-work-scalability/128 128 1024; \
	./main.out 10640 3730 500 0.001 local-work-scalability/64 64 1024; \
	./main.out 10640 3730 500 0.001 local-work-scalability/32 32 1024; \
	./main.out 10640 3730 500 0.001 local-work-scalability/16 16 1024; \


scalability-global-experiments:
	./main.out 10640 3730 500 0.001 global-work-scalability/1024 256 1024; \
	./main.out 10640 3730 500 0.001 global-work-scalability/512 256 512; \
	./main.out 10640 3730 500 0.001 global-work-scalability/256 256 256; \
	./main.out 10640 3730 500 0.001 global-work-scalability/128 256 128; \
	./main.out 10640 3730 500 0.001 global-work-scalability/64 256 64; \

	
experiments: compile scalability-size-experiments scalability-local-experiments scalability-global-experiments

documentation:
	doxygen Doxyfile
