#pragma once

#include "policy.hpp"

const double EPSILON = 1e-3;

template <typename B>
class SarsaLearner: public Policy<B> {
public:
    using A = typename Policy<B>::A;
    using Board = B;
    using S = typename Policy<B>::S;
    using SA = typename Policy<B>::SA;

    SarsaLearner(double eps, double a, double g, double l): Policy<B>(eps), Alpha(a), Gamma(g), Lambda(l) {}

    A Sample(const B &board, double rew = 0) {
        UpdatePolicy(board);
        A act = Policy<B>::Sample(board);
        SA sa = {board.Compress(), act};

        if (!first) {
            double q = qmap[sa];
            Backup(q, rew);
        } else {
            first = false;
        }

        prev_sa = sa;
        return act;
    }

    void TerminalReward(double rew) {
        if (!first) {
            Backup(0, rew);
        }
        first = true;
        zmap.clear();
    }

    using QMap = std::unordered_map<SA, double>;
    const QMap &ActValues() const {
        return qmap;
    }
private:

    void UpdatePolicy(const Board &b) {
        S s = b.Compress();

        double best = -1e9;
        A best_a;
        for (A a: GetTurns<A>()) {
            double v = qmap[{s,a}];
            if (v > best) {
                best_a = a;
                best = v;
            }
        }
        this->pmap[s] = best_a;
    }

    void Backup(double q, double r) {
        double prev_q = qmap[prev_sa];

        double delta = r + Gamma * q - prev_q;
        zmap[prev_sa] = 1;

        for (auto zit = zmap.begin(); zit != zmap.end();) {
            const SA &zsa = zit->first;
            const double z = zit->second;
            qmap[zsa] = qmap[zsa] + Alpha * z * delta;
            double new_z = Lambda * Gamma * z;
            if (new_z > EPSILON &&
                !(zsa.first == prev_sa.first && zsa.second != prev_sa.second)) { // replacing traces
                zit->second = new_z;
                zit++;
            } else {
                zit = zmap.erase(zit);
            }
        }
    }

    const double Alpha; // backup speed
    const double Gamma; // discount coefficient
    const double Lambda; // Lambda in Sarsa(L)

    bool first = true;
    SA prev_sa;
    double prev_reward;
;
    QMap qmap; // action-value
    QMap zmap; // eligibility
};
