
local M = {}

local ffi = require 'ffi'


ffi.cdef [[

    typedef struct board board;

    typedef enum turn { Up = 1, Left = 2, Down = 3, Right = 4 } turn;

    int board_size();

    board *board_new();
    void board_free(board *);
    void board_copy(board *b_dst, const board *b_src);
    void board_print(const board *b);
    bool board_move(board *b, turn t);
    bool board_random_gen(board *b);
    bool board_is_terminal(const board *b, double *r);
    unsigned long board_compress(const board *b);

    void srand(unsigned int seed);
]]

local C = ffi.C
local B = ffi.load("board.so")

-- metatable for compressed board state
local cmt = {
   __tostring = function(c)
      local s = ""
      for i=1,M.S do
         for j=1,M.S do
            local k = (i-1)*M.S + j
            local res = c:at(k)
            s = s .. tostring(res)
         end
         s = s .. "\n"
      end
      return s
   end,
   __index = {
      at = function(c,k)
         assert(k > 0)
         assert(k <= M.S*M.S)
         local res = bit.band(bit.rshift(c[1], (M.S*M.S - k)*4) , 0xf)
         return tonumber(res)
      end,
      u64 = function(c)
         return tonumber(c[1])
      end
   }
}

local c_d = ffi.new("double[1]")

local mt = {
   __index = {
      Print = function(b) B.board_print(assert(b)) end,
      RandomGen = function(b) return B.board_random_gen(assert(b)) end,
      Move = function(b,d) return B.board_move(assert(b), d) end,
      Compress = function(b) local res = {B.board_compress(assert(b))} ; setmetatable(res, cmt); return res end,
      IsTerminal = function(b) return B.board_is_terminal(assert(b), c_d), tonumber(c_d[0]) end
   }
}

local Board = ffi.metatype("board", mt)

M.new = function() return ffi.gc(B.board_new(), B.board_free) end
M.copy = B.board_copy

M.Up = C.Up
M.Down = C.Down
M.Left = C.Left
M.Right = C.Right

M.S = B.board_size()

M.srand = function(s) C.srand(s) end

return M
