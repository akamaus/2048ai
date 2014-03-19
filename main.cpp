#include <tuple>

#include "board.hpp"

using GameBoard = Board<4>;

using MoveAnalysis = std::tuple<Turn, double>;

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

uint ai(uint depth, bool display) {
    GameBoard b;
    b.RandomGen();

    while(b.NumFree() > 0) {
        b.RandomGen();
        if (display) {
            b.Print();
        }

        Turn best_turn;
        double score = get_best_move(b, depth, best_turn);
        if (score >= 0) {
            if (display) {
                std::cout << "       move " << print_turn(best_turn) << std::endl;
            }
            b.Move(best_turn);
        }
    }
    if (display) std::cout << "Turn " << b.GetTurn() << std::endl;
    return b.GetTurn();
}

const uint num_trials = 100;

int main(int argc, char *argv[]) {
    if (argc != 2) return -1;

    int depth = atoi(argv[1]);

    uint sum = 0;
    for (uint i=0; i<num_trials; i++) {
        uint res = ai(depth, false);
        std::cout << res << " ";
        sum += res;
    }
    std::cout << "\n Avg: " << (double)sum / num_trials << std::endl;
    //   interactive();
//    ai(1);
}
