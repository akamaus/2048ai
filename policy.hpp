#pragma once

#include <unordered_map>
#include <utility>

template <typename B>
class Policy {
public:
    using A = typename B::Action;
    using S = typename B::Compressed;
    using SA = std::pair<S, A>;

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

    using PMap = std::unordered_map<S,A>;
    const PMap &GetPolicy() const {
        return pmap;
    }

protected:
    int int_eps;
    PMap pmap;
};
