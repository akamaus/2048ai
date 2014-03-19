all: board.hpp main.cpp
	gcc --std=c++11 -Wall  main.cpp -lstdc++
