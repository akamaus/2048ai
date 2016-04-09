local ffi = require 'ffi'

ffi.cdef [[

    typedef struct board board;

    typedef enum turn { Up = 1, Left = 2, Down = 3, Right = 4 } turn;

    board *board_new();
    void board_print(board *b);
    bool board_move(board *b, turn t);
    bool board_random_gen(board *b);
    bool board_is_terminal(board *b, double *r);

]]

local C = ffi.C
local B = ffi.load("board.so")

local mt = {
   __index = {
      Print = function(b) B.board_print(b) end,
      RandomGen = function(b) B.board_random_gen(b) end,
      Move = function(b,d) B.board_move(b, d) end
   }
}

local Board = ffi.metatype("board", mt)

local b = B.board_new()

b:RandomGen(b)
b:Print()
print("down")
b:Move(C.Down)
b:Print()
