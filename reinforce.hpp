#include <assert.h>

#include <list>
#include <map>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <cstdlib>

using namespace std;

namespace std {

    template <>
    struct hash<Turn> {
        size_t operator()(const Turn a) const {
            return 1<< (uint)a;
        }
    };

    template <typename T1, typename T2>
    struct hash< pair<T1, T2> > {
        size_t operator()(const pair<T1,T2> &p) const {
            return hash<T1>()(p.first) ^ hash<T2>()(p.second);
        }
    };
}

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
class Policy {
public:
    using A = typename B::Action;
    using S = typename B::Compressed;
    using SA = pair<S, A>;

    static constexpr int DENOM = 1<<20;

    Policy(double eps): int_eps(DENOM * eps) {}

    A Sample(const B &b) {
        A a;
        const auto &p_act = pmap.find(b.Compress());

        if (p_act == pmap.end() || (rand() % (1 << 20)) < int_eps) {
            auto n = B::GetTurns().size();
            int k = rand() % n;
            a = B::GetTurns()[k];
        } else {
            a = p_act->second;
            goto ret;
        }
    ret:
        return a;
    }

    using PMap = unordered_map<S,A>;
    const PMap &GetPolicy() const {
        return pmap;
    }

protected:
    int int_eps;
    PMap pmap;
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

    using QMap = unordered_map< SA, Averager >;
    const QMap &ActValues() const {
        return qmap;
    }
private:
    QMap qmap;

    vector<SA> episode;
    int single_reward;
};


// class TDLearner: public Policy {
// public:

//     TDLearner(double eps, double a): Policy(eps), alpha(a) {}

//     action Sample(const Board &board) {
//         action act = Policy::Sample(board);
//         StateAct sa { board, act };

//         if (!first) {
//             const auto &prev_it = qmap.find(prev_sa);
//             const auto &it = qmap.find(sa);
//             double prev_q, q = 0;
//             if (prev_it != qmap.end()) {
//                 prev_q = prev_it->second;
//             }
//             if (it != qmap.end()) {
//                 q = it->second;
//             }

//             double new_q = prev_q + alpha * (q - prev_q);

//             if (new_q != prev_q) {
//                 qmap[prev_sa] = new_q;

//                 double best = -1e6;
//                 action best_a;
//                 bool found=false;
//                 FOR_EACH_LOOP(i,j) {
//                     action a{i,j};
//                     auto qq = qmap.find({prev_sa.first, a});
//                     if (qq != qmap.end()) {
//                         double v= qq->second;
//                         if (v > best) {
//                             best_a = a;
//                             best = v;
//                             found = true;
//                         }
//                     }
//                 }
//                 assert(found);
//                 pmap[prev_sa.first] = best_a;
//             }
//         } else {
//             first = false;
//         }

//         prev_sa = sa;
//         return act;
//     }

//     void Reward(double rew) {
//         if (!first) {
//             qmap[prev_sa] = rew;
//         }
//     }

//     void UpdatePolicy() {
//     }
//     const stateact_values &ActValues() const {
//         return qmap;
//     }
// private:
//     double alpha;
//     bool first = true;
//     StateAct prev_sa;
//     double prev_reward;

//     stateact_values qmap;
// };

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

