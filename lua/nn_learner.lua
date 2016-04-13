
local M = {}

local Board = require 'board'

-- Neural network policy
local Cells = Board.S * Board.S -- num of elems on board
local CellVars = 16 -- num of elem variants

local NN = require 'nn'

-- build Torch-NN network
local function M.build_net(w, name)
   name = name or ""
   local net = NN.Sequential()
   net:add(nn.Linear(Cells * CellVars,1))
--   net:add(nn.Tanh())
--   net:add(nn.Linear(w,1))

   net.name = string.format("%s_layer%d_%d", name, Cells * CellVars, 1)

   return net
end

-- write board state into Tensor
local function encode_state(t, st)
   t:fill(0)
   for i=1,Cells do
      for j=1,st:at(i) do
         t[(i-1) * CellVars + j] = 1
      end
   end
end

local function draw_state(t, st)
   encode_state(t, st)
   local img = t:clone()
   GP.imagesc(img:resize(Cells, CellVars))
end

local mt = {
   __index = {
      -- returns estimated value of a position
      est_value = function(L, st)
         encode_state(L.in_t, st)
         local val = L.net:forward(L.in_t)
         return val[1]
      end,
      -- learns true value and returns error
      learn = function(L, st, val)
         encode_state(L.in_t, st)
         L.out_t[1] = val
         local pred = L.net:forward(L.in_t)
         L.net:backward(L.in_t, L.cr:backward(pred, L.out_t))
         return pred[1] - val
      end,
      -- applies learned material
      apply = function(L, rate)
         L.net:updateParameters(rate)
         L.net:zeroGradParameters()
      end
   }
}

local function M.build_nn_learner(L, w, name)
   local L = L or {
      name = name,
      net = build_net(w, name),
      in_t = torch.Tensor(Cells * CellVars),
      out_t = torch.Tensor(1),
      cr = nn.MSECriterion()
   }
   L.net:zeroGradParameters()

   setmetatable(L, mt)
   return L
end

return M
