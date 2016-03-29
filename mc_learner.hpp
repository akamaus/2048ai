#pragma once

#include "policy.hpp"

class Averager {
public:
    Averager(): avg(0), n(0) {}
    void Add(double x) {
        n++;
        avg = avg + (x - avg) / n;
    }
    double Avg() const {
        return avg;
    }
private:
    double avg;
    int n;
};


template <typename B>
class MCLearner: public Policy<B> {
public:
    using A = typename Policy<B>::A;
    using Board = B;
    using S = typename Policy<B>::S;
    using SA = typename Policy<B>::SA;

    MCLearner(double eps): Policy<B>(eps) {}

    A Sample(const B &b) {
        A a = Policy<B>::Sample(b);
        episode.push_back({b.Compress(),a});
        return a;
    }

    void Reward(int rew) {
        single_reward = rew;
    }
    void UpdatePolicy() {
        // Updating action values
        for (const auto &sa : episode) {
            qmap[sa].Add(single_reward);
        }
        // Updating policy
        for (const auto &sa : episode) {
            const S &sb = sa.first;
            double best = -1e6;
            A best_a;
            bool found=false;
            for (A a : B::GetTurns()) {
                auto q = qmap.find({sb, a});
                if (q != qmap.end()) {
                    double v= q->second.Avg();
                    if (v > best) {
                        best_a = a;
                        best = v;
                        found = true;
                    }
                }
            }
            assert(found);
            this->pmap[sb] = best_a;
        }
        episode.clear();
        single_reward = 0;
    }

    using QMap = std::unordered_map< SA, Averager >;
    const QMap &ActValues() const {
        return qmap;
    }
private:
    QMap qmap;

    std::vector<SA> episode;
    int single_reward;
};
