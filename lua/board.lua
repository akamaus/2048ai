
local M = {}
M.S = 4

local ffi = require 'ffi'


ffi.cdef [[

    typedef struct board board;

    typedef enum turn { Up = 1, Left = 2, Down = 3, Right = 4 } turn;

    board *board_new();
    void board_print(const board *b);
    bool board_move(board *b, turn t);
    bool board_random_gen(board *b);
    bool board_is_terminal(const board *b, double *r);
    unsigned long board_compress(const board *b);


]]

local C = ffi.C
local B = ffi.load("board.so")

-- metatable for compressed board state
local cmt = {
   __tostring = function(c)
      local s = ""
      for i=0,M.S-1 do
         for j=0,M.S-1 do
            local b = c[1]
            local k = i*M.S + j
            local res = bit.band(bit.rshift(b, 60 - k*4), 0xf)
            s = s .. tonumber(res)
         end
         s = s .. "\n"
      end
      return s
   end
}

local mt = {
   __index = {
      Print = function(b) B.board_print(b) end,
      RandomGen = function(b) return B.board_random_gen(b) end,
      Move = function(b,d) return B.board_move(b, d) end,
      Compress = function(b) local res = {B.board_compress(b)} ; setmetatable(res, cmt); return res end
   }
}

local Board = ffi.metatype("board", mt)

M.new = B.board_new
M.Up = C.Up
M.Down = C.Down
M.Left = C.Left
M.Right = C.Right

return M
