#include "../board.hpp"

extern "C" {
    typedef Board<4> board;
    enum turn { Up = 1, Left = 2, Down = 3, Right = 4 };

    board *board_new();
    void board_print(board *b);
    bool board_move(board *b, turn t);
    bool board_random_gen(board *b);
    bool board_is_terminal(board *b, double *r);

    board *board_new() {
        return new board();
    }

    void board_print(board *b) {
        b->Print();
    }

    bool board_move(board *b, turn t) {
        return b->Move((Turn)t);
    }

    bool board_random_gen(board *b) {
        return b->RandomGen();
    }

    bool board_is_terminal(board *b, double *r) {
        return b->IsTerminal(*r);
    }
};

