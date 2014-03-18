#include <tuple>

#include "board.hpp"

using GameBoard = Board<4>;

using MoveAnalysis = std::tuple<Turn, double>;

MoveAnalysis get_worst_rnd(GameBoard board, Turn t, uint depth);
MoveAnalysis get_best_move(GameBoard board, uint k, uint depth);
double eval_board(const GameBoard &board);

MoveAnalysis get_worst_rnd(GameBoard board, Turn t, uint depth) {
    board.Move(t);
    if (depth == 0) {
        return std::make_tuple(t, eval_board(board));
    }
    depth--;

    MoveAnalysis worst { Turn::Left, 1e10};
    for (uint k=0; k<board.NumFree(); k++) {
        auto result = get_best_move(board, k, depth);
        if (std::get<1>(worst) > std::get<1>(result)) {
            worst = result;
        }
    }
    return worst;
}

MoveAnalysis get_best_move(GameBoard board, uint k, uint depth) {
    static std::array<Turn,4> moves = {Turn::Left, Turn::Right, Turn::Up, Turn::Down};
    board.Gen(k);

    MoveAnalysis best { Turn::Left, -1 };
    for (auto next_turn : moves) {
        auto result = get_worst_rnd(board, next_turn, depth);
        if (std::get<1>(best) < std::get<1>(result)) {
            best = result;
        }
    }
    return best;
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
}

void ai() {
    GameBoard b;
    b.RandomGen();

    while(b.NumFree() > 0) {
        uint k = rand() % b.NumFree();

        auto res = get_best_move(b, k, 1);
        b.Gen(k);
        b.Print(); std::cout << "       move " << print_turn(std::get<0>(res)) << std::endl;
        b.Move(std::get<0>(res));
        b.Print(); std::cout << std::endl << std::endl;
    }
}

int main() {
    //   interactive();
    ai();
}
