#include <map>
#include <functional>
#include <tuple>
#include <vector>
#include <utility>


#include "board.hpp"
#include "reinforce.hpp"
//#include "mc_learner.hpp"
#include "sarsa_learner.hpp"
#include "q_learner.hpp"
#include "test_pole.hpp"


using GameBoard = Board<4>;

using MoveAnalysis = std::tuple<Turn, double>;
using Strategy = std::function<Turn(const GameBoard &)>;

MoveAnalysis get_worst_rnd(GameBoard board, Turn t, uint depth);
MoveAnalysis get_best_move(GameBoard board, uint k, uint depth);
double eval_board(const GameBoard &board);

std::string print_turn(Turn t);

/*
MoveAnalysis get_worst_rnd(GameBoard board, Turn t, uint depth) {
    board.Move(t);
    if (depth == 0) {
        return std::make_tuple(t, eval_board(board));
    }
    depth--;

    if (board.NumFree() == 0) {
        return std::make_tuple(Turn::Left, -1000);
    }

    MoveAnalysis worst { Turn::Left, 1e10};
    for (uint k=0; k<board.NumFree(); k++) {
        auto result = get_best_move(board, k, depth);
        if (std::get<1>(worst) > std::get<1>(result)) {
            worst = result;
        }
    }
    return worst;
}
*/

double get_best_move(const GameBoard &board, uint depth, Turn &ret_turn) {
    static std::array<Turn,4> moves = {Turn::Left, Turn::Right, Turn::Up, Turn::Down};

    if(depth == 0) {
        return eval_board(board);
    }

    depth--;

    double best_score = -1;
    Turn best_turn;
    for (auto turn : moves) {
        GameBoard b(board); Turn t;

        if (b.Move(turn)) {
            if (!b.RandomGen()) continue;
            double score = get_best_move(b, depth,t);
            if (score > best_score) {
                best_score = score;
                best_turn = turn;
            }
        }
    }
    ret_turn = best_turn;
    return best_score;
}

double eval_board(const GameBoard &board) {
    return board.NumFree();
}

void interactive() {
    GameBoard b;

    char c;
    do {
        b.Print();
        std::cout << std::endl;

        c = std::cin.get();
        switch(c) {
        case 'a':
            b.Move(Turn::Left);
            break;
        case 'd':
            b.Move(Turn::Right);
            break;
        case 'w':
            b.Move(Turn::Up);
            break;
        case 's':
            b.Move(Turn::Down);
            break;
        case 'r':
            b.RandomGen();
            break;
        case 'q':
            return;
        default:
            continue;
        };

//        std::cout << c << std::endl;
    } while (true);
}

std::string print_turn(Turn t) {
    switch(t) {
    case Turn::Left:
        return std::string{"Left"};
    case Turn::Right:
        return std::string{"Right"};
    case Turn::Up:
        return std::string{"Up"};
    case Turn::Down:
        return std::string{"Down"};
    }
    return "UNKNOWN";
}

Strategy make_ai(uint depth) {
    Strategy s = [depth](const GameBoard &b0) {
        Turn best_turn = Turn::Left;
//        double score = 
        get_best_move(b0, depth, best_turn);
        return best_turn;
    };

    return s;
}

Strategy random_strategy = [](const GameBoard &b0) {
    return static_cast<Turn>(rand()%4);
};

std::vector<GameBoard> gather_data(Strategy strat) {
    const uint num_trials = 100;

    std::vector<GameBoard> results;
    for (uint i=0; i< num_trials; i++) {
        GameBoard b;
        b.RandomGen();

        while(b.NumFree() > 0) {
            b.RandomGen();

            Turn t = strat(b);
//            std::cout << print_turn(t);
            b.Move(t);
        }
        std::cerr << ".";
        results.push_back(b);
    }
    return results;
}

void print_statistics(const std::vector<GameBoard> &results) {
    uint sum = 0;
    uint best = 0;
    uint worst = 1000000;

    std::map<uint, uint> best_tiles;

    for (const auto &rb : results) {
        uint score = rb.GetTurn();

        if (score > best) {
            best = score;
        } else if (score < worst) {
            worst = score;
        }
        sum += score;

        uint best = rb.BestTile();
        if (best_tiles.count(best) == 0) {
            best_tiles.insert(std::make_pair(best, 1));
        } else {
            best_tiles[best]++;
        }
    }
    std::cout << "Best: " << best << std::endl;
    std::cout << "Worst: " << worst << std::endl;
    std::cout << "Avg: " << (double)sum / results.size() << std::endl << std::endl;

    for (auto it : best_tiles) {
        std::cout << "[" << it.first << "] = " << it.second << std::endl;
    }
}

void Usage() {
    std::cerr << "Usage: ./ai2048 TST_SARSA <num_trials> <eps> <alpha> <gamma> <lambda>" << std::endl;
    std::cerr << "Usage: ./ai2048 SARSA <num_trials> <eps> <alpha> <gamma> <lambda>" << std::endl;
    std::cerr << "Usage: ./ai2048 MC <num_trials>" << std::endl;
    std::cerr << "Usage: ./ai2048 MM <depth>" << std::endl;
    std::cerr << "Usage: ./ai2048 I" << std::endl;

    exit(1);
}

void run_minimax(int depth) {
    Strategy s;
    if (depth > 0) {
        s = make_ai(depth);
    } else {
        s = random_strategy;
    }

    auto results = gather_data(s);
    print_statistics(results);
}

void run_mc_learner(int num_trials, double eps) {
//    MCLearner<GameBoard> learner(eps);
//    driver(num_trials, learner);
}

#include <fstream>
#include <string>

template <>
void visualize_learner<QLearner<TestPole>, TestPole>(const QLearner<TestPole> &p, const TestPole &b) {
    static int f_num;
    f_num++;
    std::ofstream f_q(std::string("plots/q_") + std::to_string(f_num) + ".mat");
    std::ofstream f_p(std::string("plots/p_") + std::to_string(f_num) + ".mat");
//    auto &f_q(std::cout);

    for (int y=1; y<=b.sy; y++) {
        for (int x=1; x<=b.sx; x++) {
            auto it = p.GetPolicy().find({x,y});
            int ti = 0;
            double q = 0;
            if (it != p.GetPolicy().end()) {
                Turn t = it->second;
                ti = (int)t;
                auto qit = p.ActValues().find({{x,y},t});
                if (qit != p.ActValues().end()) {
                    q = qit->second;
                }
            }
            if (ti != 0) {
                f_p << ti << " ";
            } else {
                f_p << '?' << " ";
            }
            if (q != 0) {
                f_q << q << " ";
            } else {
                f_q << '?' << " ";
            }
        }
        f_p << std::endl;
        f_q << std::endl;
    }
}

template <>
void visualize_learner<SarsaLearner<TestPole>, TestPole>(const SarsaLearner<TestPole> &p, const TestPole &b) {
    static int f_num;
    f_num++;
    std::ofstream f_q(std::string("plots/q_") + std::to_string(f_num) + ".mat");
    std::ofstream f_p(std::string("plots/p_") + std::to_string(f_num) + ".mat");
//    auto &f_q(std::cout);

    for (int y=1; y<=b.sy; y++) {
        for (int x=1; x<=b.sx; x++) {
            auto it = p.GetPolicy().find({x,y});
            int ti = 0;
            double q = 0;
            if (it != p.GetPolicy().end()) {
                Turn t = it->second;
                ti = (int)t;
                auto qit = p.ActValues().find({{x,y},t});
                if (qit != p.ActValues().end()) {
                    q = qit->second;
                }
            }
            if (ti != 0) {
                f_p << ti << " ";
            } else {
                f_p << '?' << " ";
            }
            if (q != 0) {
                f_q << q << " ";
            } else {
                f_q << '?' << " ";
            }
        }
        f_p << std::endl;
        f_q << std::endl;
    }
}

template <>
void visualize_learner<SarsaLearner<GameBoard>, GameBoard>(const SarsaLearner<GameBoard> &p, const GameBoard &b) {}


void test_q_learner(int num_trials, double eps, double alpha, double gamma) {
//    QLearner<TestPole> learner(eps, alpha, gamma);
//    driver(num_trials, learner);
}

int main(int argc, char *argv[]) {
    if (argc < 2) Usage();

    std::string mode = argv[1];
    if (mode == "I") {
        interactive();
    } else {
        if (mode == "MC") {
            if (argc != 4) Usage();
            int trials = atoi(argv[2]);
            double eps = atoi(argv[3]);
            run_mc_learner(trials, eps);
        } else if (mode == "SARSA") {
            if (argc != 7) Usage();
            int trials = std::stoi(argv[2]);
            double eps = std::stod(argv[3]);
            double alpha = std::stod(argv[4]);
            double gamma = std::stod(argv[5]);
            double lambda = std::stod(argv[6]);
            SarsaLearner<GameBoard> learner(eps, alpha, gamma, lambda);
            driver(trials, learner);
        } else if (mode == "TST_Q") {
            if (argc != 6) Usage();
            int trials = std::stoi(argv[2]);
            double eps = std::stod(argv[3]);
            double alpha = std::stod(argv[4]);
            double gamma = std::stod(argv[5]);
            test_q_learner(trials, eps, alpha, gamma);
        } else if (mode == "TST_SARSA") {
            if (argc != 7) Usage();
            int trials = std::stoi(argv[2]);
            double eps = std::stod(argv[3]);
            double alpha = std::stod(argv[4]);
            double gamma = std::stod(argv[5]);
            double lambda = std::stod(argv[6]);

            SarsaLearner<TestPole> learner(eps, alpha, gamma, lambda);
            driver(trials, learner);
        } else if (mode == "MM") {
            int depth = atoi(argv[2]);
            run_minimax(depth);
        } else Usage();
    }

    return 0;
}
