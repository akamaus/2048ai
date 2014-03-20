#include <map>
#include <functional>
#include <tuple>
#include <vector>
#include <utility>


#include "board.hpp"



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

const uint num_trials = 100;

Strategy make_ai(uint depth) {
    Strategy s = [depth](const GameBoard &b0) {
        Turn best_turn = Turn::Left;
        double score = get_best_move(b0, depth, best_turn);
        return best_turn;
    };

    return s;
}

Strategy random_strategy = [](const GameBoard &b0) {
    return static_cast<Turn>(rand()%4);
};

std::vector<GameBoard> gather_data(Strategy strat) {
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

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: ./ai2048 <depth>" << std::endl;
        return -1;
    }

    int depth = atoi(argv[1]);

    uint sum = 0;
    uint best = 0;
    uint num_wins = 0;

    Strategy s;
    if (depth > 0) {
        s = make_ai(depth);
    } else {
        s = random_strategy;
    }

    auto results = gather_data(s);
    print_statistics(results);

    return 0;
}
