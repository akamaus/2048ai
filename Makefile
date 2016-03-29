CFLAGS= --std=c++11 -Wall -O3 -g

ai2048: board.hpp reinforce.hpp main.cpp
	gcc $(CFLAGS) main.cpp -lstdc++ -o ai2048
