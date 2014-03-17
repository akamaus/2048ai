#include <array>
#include <iostream>

#include <cassert>

enum class Turn { Left, Right, Up, Down };

typedef uint8_t value;

template <uint size>
class Board {
public:
    Board(): _board {0} {}

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

    value &At(uint i, uint j) {
        assert(i<size);
        assert(j<size);
        return _board[i*size + j];
    }

    void Move(Turn t) {
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
                    if (next == 0)
                        continue;
                    if (cur == 0 && next != 0) {
                        cur = next;
                        next = 0;
                        changed |= true;
                        continue;
                    }
                    if (cur != 0 && cur == next) {
                        cur++;
                        next = 0;
                        changed |= true;
                        continue;
                    }
                }
            }
            if (!changed)
                break;

        }
    }

    std::array<value, size*size> _board;
};

