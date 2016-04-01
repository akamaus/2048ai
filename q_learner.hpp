#pragma once

#include "policy.hpp"


template <typename B>
class QLearner: public Policy<B> {
public:
    using A = typename Policy<B>::A;
    using Board = B;
    using S = typename Policy<B>::S;
    using SA = typename Policy<B>::SA;

    QLearner(double eps, double a, double g): Policy<B>(eps), Alpha(a), Gamma(g) {}

    A Sample(const B &board) {
        prev_act = Policy<B>::Sample(board);
        prev_sa = { board.Compress(), prev_act };

        return prev_act;
    }

    void Reward(const B &board, double r) {
        S s = board.Compress();

        const auto &it = qmap.find(prev_sa);
        double prev_q = 0;
        if (it != qmap.end()) {
            prev_q = it->second;
        }

        double best = -1e9;
        A best_a;
        bool found=false;
        for (A a: GetTurns<A>()) {
            auto qq = qmap.find({s, a});
            if (qq != qmap.end()) {
                double v= qq->second;
                if (v > best) {
                    best_a = a;
                    best = v;
                    found = true;
                }
            }
        }
        if (!found) best = 0;

        double new_q = prev_q + Alpha * ( r + Gamma * best - prev_q );
        if (new_q != prev_q) {
            qmap[prev_sa] = new_q;
        }

        int new_p = UpdatePolicy(prev_sa.first);

//        B::Print(prev_sa.first); B::Print(s);
//        std::cout << "r=" << r << ";pq=" << prev_q << ";q=" << new_q << ";p=" << (int)new_p << std::endl;

    }

    int UpdatePolicy(const S &s) {
        double best = -1e9;
        A best_a = Turn::Left;
        bool found=false;

        for (A a: GetTurns<A>()) {
            auto qq = qmap.find({s, a});
            double v=0;
            if (qq != qmap.end()) {
                v = qq->second;
            }
            if (v > best) {
                best_a = a;
                best = v;
                found = true;
            }
        }
        this->pmap[s] = best_a;

        return found?(int)best_a:0;
    }

    using QMap = std::unordered_map<SA, double>;
    const QMap &ActValues() const {
        return qmap;
    }
private:
    const double Alpha;
    const double Gamma;

    bool first = true;
    SA prev_sa;
    A prev_act;
;
    QMap qmap;
};
