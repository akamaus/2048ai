#pragma once

template <typename B>
class TDLearner: public Policy<B> {
public:
    using A = typename Policy<B>::A;
    using Board = B;
    using S = typename Policy<B>::S;
    using SA = typename Policy<B>::SA;

    TDLearner(double eps, double a): Policy<B>(eps), alpha(a) {}

    A Sample(const B &board) {
        A act = Policy<B>::Sample(board);
        SA sa { board.Compress(), act };

        if (!first) {
            const auto &prev_it = qmap.find(prev_sa);
            const auto &it = qmap.find(sa);
            double prev_q, q = 0;
            if (prev_it != qmap.end()) {
                prev_q = prev_it->second;
            }
            if (it != qmap.end()) {
                q = it->second;
            }

            double new_q = prev_q + alpha * (q - prev_q);

            if (new_q != prev_q) {
                qmap[prev_sa] = new_q;

                double best = -1e6;
                A best_a;
                bool found=false;
                for (A a: B::GetTurns()) {
                    auto qq = qmap.find({prev_sa.first, a});
                    if (qq != qmap.end()) {
                        double v= qq->second;
                        if (v > best) {
                            best_a = a;
                            best = v;
                            found = true;
                        }
                    }
                }
                assert(found);
                this->pmap[prev_sa.first] = best_a;
            }
        } else {
            first = false;
        }

        prev_sa = sa;
        return act;
    }

    void Reward(double rew) {
        if (!first) {
            qmap[prev_sa] = rew;
        }
        first = true;
    }

    void UpdatePolicy() {
        // proceeds inline in Sample()
    }

    using QMap = std::unordered_map<SA, double>;
    const QMap &ActValues() const {
        return qmap;
    }
private:
    double alpha;

    bool first = true;
    SA prev_sa;
    double prev_reward;
;
    QMap qmap;
};
