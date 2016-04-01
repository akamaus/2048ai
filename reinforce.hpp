#pragma once

#include <assert.h>

#include <list>
#include <map>
#include <vector>
#include <iostream>
#include <cstdlib>

#include "board.hpp"

using namespace std;

template <typename L, typename B>
void visualize_learner(const L &, const B &);

template <typename L>
void driver(long num_episodes, L &player) {
    double avg_reward = 0;

    for (long k=0; k < num_episodes; k++) { // episode
        typename L::Board b;
        double t_reward = 0;

        double s_reward = 0;
        while(!b.IsTerminal(t_reward)) {
            b.EnvTurn();

            typename L::A act = player.Sample(b, s_reward);
//            b.Print();
//            std::cout << "act:" << (int)act << std::endl;
            s_reward = b.Move(act);
//            t_reward += s_reward;
        }
        player.Reward(t_reward);
        player.UpdatePolicy();
        visualize_learner(player, b);

        avg_reward = avg_reward * 0.99 + t_reward * 0.01;
//        if (true || k % 100 == 0)
            printf("n %lu q %lu, p %lu r %f ar %f\n", k, player.ActValues().size(), player.GetPolicy().size(), t_reward, avg_reward);
        fflush(stdout);
    }
}

