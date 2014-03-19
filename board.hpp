#include <stdlib.h>

#include <array>
#include <iostream>
#include <cassert>



enum class Turn { Left, Right, Up, Down };

typedef uint8_t value;

template <uint size>
class Board {
    const static value new_cell = 1;
public:
    Board(): _board{0}, _board_merges{0}, _num_free(size*size), _cur_turn{0} {}

    void Print() {
        for (int i=0; i<size; i++) {
            for (int j=0; j<size; j++) {
                uint c = At(i,j);
                if (c == 0)
                    std::cout << '.';
                else
                    std::cout << static_cast<char>('A' + c - 1);
            }
            std::cout << std::endl;
        }
    }
private:
    value &At(uint i, uint j) {
        assert(i<size);
        assert(j<size);
        return _board[i*size + j];
    }

    bool MergedAt(uint i, uint j) {
        return _board_merges[i*size + j] == _cur_turn;
    }

    void MergeAt(uint i, uint j) {
        _board_merges[i*size + j] = _cur_turn;
    }
public:
    bool Move(Turn t) {
        _cur_turn++;
        bool changed = false;
        for (uint k=0; k<size; k++) {
            for (uint m=0; m<size; m++) {
                for (uint n=0; n<size-1; n++) {
                    uint cur_i, cur_j, next_i, next_j;
                    switch(t) {
                    case Turn::Left:
                        cur_i = m; cur_j = n;
                        next_i = m, next_j = n+1;
                        break;
                    case Turn::Right:
                        cur_i = m; cur_j = size-1-n;
                        next_i = m, next_j = size-1-(n+1);
                        break;
                    case Turn::Up:
                        cur_i = n; cur_j = m;
                        next_i = n+1, next_j = m;
                        break;
                    case Turn::Down:
                        cur_i = size-1-n; cur_j = m;
                        next_i = size-1-(n+1), next_j = m;
                        break;
                    };
                    value &cur(At(cur_i, cur_j));
                    value &next(At(next_i, next_j));
                    if (next == 0) // nothing to move
                        continue;
                    if (cur == 0 && next != 0) { // move
                        cur = next;
                        next = 0;
                        changed |= true;
                        continue;
                    }
                    if (cur != 0 && cur == next && !MergedAt(cur_i, cur_j) && !MergedAt(next_i, next_j)) { // merge
                        cur++;
                        next = 0;
                        changed |= true;
                        MergeAt(cur_i, cur_j);
                        _num_free++;
                        continue;
                    }
                }
            }
            if (!changed) {
                if (k == 0) {
                    return false;
                }
                break;
            }
        }
        return true;
    }

    void Gen(uint nth) {
        assert(nth < _num_free);

        for (int i=0; i<size; i++) {
            for (int j=0; j<size; j++) {
                if (At(i,j) == 0) {
                    if (nth == 0) {
                        At(i,j) = Board::new_cell;
                        _num_free--;
                        return;
                    }
                    else
                        nth--;
                }
            }
        }
    }

    void RandomGen() {
        if (_num_free == 0)
            return;

        uint nth = rand() % _num_free;
        Gen(nth);
    }

    uint NumFree() const {
        return _num_free;
    }
    uint GetTurn() const {
        return _cur_turn;
    }

private:
    std::array<value, size*size> _board; // a board
    std::array<value, size*size> _board_merges; // a mask of recent merges
    uint _num_free; // number of free cells
    uint _cur_turn; // turn number
};
