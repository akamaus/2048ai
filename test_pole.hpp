#pragma once

class TestPole {
public:
    using Compressed = std::pair<int,int>;
    using Action = Turn;

    TestPole(): TestPole(50,50) {}
    TestPole(int _sx, int _sy):sx(_sx), sy(_sy) {
        pos_x = sx/4;
        pos_y = sy/4;
    }

    void EnvTurn() {}
    double Move(Turn t) {
        switch(t) {
        case Turn::Left:
            pos_x = std::max(pos_x-1,1);
            break;
        case Turn::Right:
            pos_x = std::min(pos_x+1,sx);
            break;
        case Turn::Up:
            pos_y = std::max(pos_y-1,1);
            break;
        case Turn::Down:
            pos_y = std::min(pos_y+1,sy);
            break;
        };

        return -1;
    }

    bool IsTerminal(double &reward) const {
        if ((pos_x < 3 * sx / 4) && (pos_y >= sy / 2) && (pos_y <= 2 * sy / 3 )) {
            reward = -50;
            return true;
        }
        // if (pos_x==1 && pos_y == 1) {
        //     reward += -100;
        //     return true;
        // }
        if (pos_x==1 && pos_y == sy-1) {
            reward = 200;
            return true;
        }
        return false;
    }

    Compressed Compress() const {
        return {pos_x, pos_y};
    }

    void Print() const {
        cout << "<" << pos_x << ";" << pos_y << ">";
    }

    static void Print(Compressed &c) {
        cout << "<" << c.first << ";" << c.second << ">";
    }

    uint BestTile() { return -1; };
    const int sx, sy;

private:
    int pos_x, pos_y;
};
