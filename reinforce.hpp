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

constexpr int skip_k = 10000;
constexpr double avg_k = 1.0 / skip_k;

template <typename L>
void driver(long num_episodes, L &player) {
    double avg_reward = 0;

    for (long k=0; k < num_episodes; k++) { // episode
        double min_reward;
        double max_reward;

        typename L::Board b;
        double total_reward = 0;
        double t_reward = 0;
        double s_reward = 0;
        while(!b.IsTerminal(t_reward)) {
            b.EnvTurn();

            typename L::A act = player.Sample(b, s_reward);
            s_reward = b.Move(act);
            total_reward += s_reward;
        }
        player.TerminalReward(s_reward + t_reward);
        total_reward += t_reward;
//        player.UpdatePolicy();

        min_reward = std::min(min_reward, total_reward);
        max_reward = std::max(max_reward, total_reward);

        
        avg_reward = avg_reward * (1-avg_k) + total_reward * avg_k;
        if (k % skip_k == 0) {
            printf("n %lu q %lu, p %lu r %f ar %f M %d mr %f Mr %f\n", k, player.ActValues().size(), player.GetPolicy().size(), total_reward, avg_reward, b.BestTile(), min_reward, max_reward);
            visualize_learner(player, b);

            min_reward = 1e9;
            max_reward = -1e9;
        }

        fflush(stdout);
    }
}

