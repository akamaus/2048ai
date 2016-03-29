#pragma once

#include <assert.h>

#include <list>
#include <map>
#include <vector>
#include <iostream>
#include <cstdlib>

#include "board.hpp"

using namespace std;



template <typename L>
void driver(int num_episodes, L &player) {
    double avg_reward = 0;

    for (int k=0; k < num_episodes; k++) { // episode
        typename L::Board b;

        int reward = 0;
        while(b.NumFree() > 0) {
            b.RandomGen();
            typename L::A act = player.Sample(b);
            b.Move(act);
//            b.Print();
//            std::cout << std::endl;
        }
        reward = b.GetTurn();

        player.Reward(reward);
        player.UpdatePolicy();

        avg_reward = avg_reward * 0.99 + reward * 0.01;

        printf("q %lu, p %lu r %d ar %f\n", player.ActValues().size(), player.GetPolicy().size(), reward, avg_reward);
    }
}

