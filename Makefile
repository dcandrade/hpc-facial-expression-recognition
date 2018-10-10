all: compile run

compile:
	g++ -O2 main.cpp -o main.out

run:
	./main.out

debug:
	g++ -g main.cpp -o main.out
	gdb main.out

clean:
	rm *.out

profile: compile
	perf record ./main.out

run-experiments: compile
	perf record ./main.out 8092 2012 100 0.01
	mv perf.data perf_exp1.data
