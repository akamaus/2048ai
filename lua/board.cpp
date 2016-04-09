#include "../board.hpp"

extern "C" {
    typedef Board<4> board;
    enum turn { Up = 1, Left = 2, Down = 3, Right = 4 };

    board *board_new();
    void board_copy(board *b_dst, const board *b_src);
    void board_print(const board *b);
    bool board_move(board *b, turn t);
    bool board_random_gen(board *b);
    bool board_is_terminal(const board *b, double *r);
    unsigned long board_compress(const board *b);

    board *board_new() {
        return new board();
    }

    void board_copy(board *b_dst, const board *b_src) {
        *b_dst = *b_src;
    }

    void board_print(const board *b) {
        b->Print();
    }

    bool board_move(board *b, turn t) {
        return b->Move((Turn)t);
    }

    bool board_random_gen(board *b) {
        return b->RandomGen();
    }

    bool board_is_terminal(const board *b, double *r) {
        return b->IsTerminal(*r);
    }

    unsigned long board_compress(const board *b) {
        return b->Compress();
    }
};

