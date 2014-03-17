#include "board.hpp"

int main() {
    Board<4> b;
    b.At(1,1) = 3;
    b.At(3,1) = 3;
    b.At(0,0) = 4;

    b.Print();

    char c;
    do {
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
        case 'q':
            ;
        default:
            continue;
        };

        b.Print();
        std::cout << std::endl;
//        std::cout << c << std::endl;
    } while (c != 'q');


}
