all: board.hpp main.cpp
	gcc --std=c++11 -Wall -O3  main.cpp -lstdc++
