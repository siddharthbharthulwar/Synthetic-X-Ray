all: *.cpp
	g++ -Wall -std=c++11 -o lungnodulesynthesizer $^ -pthread
