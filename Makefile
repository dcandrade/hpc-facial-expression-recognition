all: compile run
compile:
	g++ main.cpp -o main.out
run:
	./main.out
debug:
	g++ -g main.cpp -o main.out
	gdb main.out
clean:
	rm *.out