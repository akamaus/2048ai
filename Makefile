CFLAGS= --std=c++11 -Wall -O3 -g

ai2048: board.hpp reinforce.hpp main.cpp sarsa_learner.hpp mc_learner.hpp reinforce.hpp policy.hpp test_pole.hpp q_learner.hpp Makefile
	gcc $(CFLAGS) main.cpp -lstdc++ -o ai2048
