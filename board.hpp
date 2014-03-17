#include <array>
#include <iostream>

template <uint size>
class Board {
public:
    Board() {}

    Print() {
        for (int i=0; i<size; i++) {
            for (int j=0; j<size; j++) {
                uint c = At(i,j);
                if (c == 0)
                    std::cout << '.';
                else
                    std::cout << 'A' + c - 2;
            }
            std::cout << endl;
        }
    }

    uint &At(uint i, uint j) {
        assert(i<size);
        assert(j<size);
        return _board[i*size + j];
    }

    std::array<uint, size*size> _board;
};

