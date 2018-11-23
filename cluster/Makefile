all: compile run
compile:
	mpic++ -O3 -fopenmp  main.cpp -o main.out


run:
	mpirun -np 3 -machinefile Machinefile /home/equipec/HPC/cluster/main.out
	#mpirun -np 3 -machinefile Machinefile /home/equipec/HPC/cluster/main.out 15640 3730 500 0.001 epoch-scalability/500

debug:
	g++ -g main.cpp -o main.out
	gdb main.out

clean:
	rm *.out
	rm -rf results
	rm perf.*

profile: 
	g++  -g -pg -O3 -o main.out main.cpp
	mpirun -np 3 -machinefile Machinefile /home/equipec/HPC/cluster/main.out
	gprof main.out > profile.txt

setup-experiments:
	mkdir results/
	mkdir results/epoch-scalability results/size-scalability
	cd results/epoch-scalability &&	mkdir 1000 500 250 && cd ../..
	cd results/size-scalability && mkdir 15640 11730 5865 && cd ../..

scalability-epoch-experiments:
	mpirun -np 3 -machinefile Machinefile /home/equipec/HPC/cluster/main.out 15640 3730 1000 0.001 epoch-scalability/1000; \
	mpirun -np 3 -machinefile Machinefile /home/equipec/HPC/cluster/main.out 15640 3730 500 0.001 epoch-scalability/500; \
	mpirun -np 3 -machinefile Machinefile /home/equipec/HPC/cluster/main.out 15640 3730 250 0.001 epoch-scalability/250; \

scalability-size-experiments:
	mpirun -np 3 -machinefile Machinefile /home/equipec/HPC/cluster/main.out 15640 3730 500 0.001 size-scalability/15640; \
	mpirun -np 3 -machinefile Machinefile /home/equipec/HPC/cluster/main.out 11730 3730 500 0.001 size-scalability/11730; \
	mpirun -np 3 -machinefile Machinefile /home/equipec/HPC/cluster/main.out 7280 3730 500 0.001 size-scalability/5865; \

	
experiments: compile scalability-size-experiments

documentation:
	doxygen Doxyfile
