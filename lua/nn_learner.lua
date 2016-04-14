
local GP = require 'gnuplot'

require 'helpers'
local Board = require 'board'
local NN = require 'nn'

local M = {}
-- Neural network policy
local Cells = Board.S * Board.S -- num of elems on board
local CellVars = 16 -- num of elem variants
local OutVars = 100

-- build Torch-NN network
local function build_net(name, w)
   name = name or ""
   local net = NN.Sequential()
   local first_layer = Cells * CellVars
   local last_layer = first_layer
   local formula = tostring(last_layer)
   for i, sz in pairs(w) do
      net:add(nn.Linear(last_layer,sz))
      net:add(nn.Tanh())
      last_layer = sz
      formula = formula .. ' ' .. tostring(sz)
   end
   net:add(nn.Linear(last_layer,OutVars))
   formula = formula .. ' ' .. OutVars

   net.name = string.format("%s_layers%s", name, formula)

   return net
end

-- write board state into Tensor
local function encode_input(t, st)
   t:fill(0)
   for i=1,Cells do
      for j=1,st:at(i) do
         t[(i-1) * CellVars + j] = 1
      end
   end
end

-- set output neuron
local function encode_output(t, val)
   for i=1,OutVars do
      local dist = math.abs(val - i)
      t[i] = -dist*math.sqrt(dist)
   end
end

-- get most active output neuron
local function decode_output(t)
   val,ind = torch.max(t,1)
   return ind[1]-1, val
end

local function draw_input(t, st)
   local img = t:clone()
   GP.imagesc(img:resize(Cells, CellVars))
end

local mt = {
   __index = {
      -- returns estimated value of a position
      est_value = function(L, st)
         encode_input(L.in_t, st)
         local pred = L.net:forward(L.in_t)
         return decode_output(pred)
      end,
      -- learns true value and returns error
      learn = function(L, st, val)
         encode_input(L.in_t, st)
         encode_output(L.out_t, val)
         local pred = L.net:forward(L.in_t)
         local pred_val = decode_output(pred)
         L.net:backward(L.in_t, L.cr:backward(pred, L.out_t))
         return pred_val - val, pred
      end,
      -- applies learned material
      apply = function(L, rate)
         L.net:updateParameters(rate)
         L.net:zeroGradParameters()
      end
   }
}

function M.build_nn_learner(name, w, L)
   local L = L or {
      name = name,
      net = build_net(name, w),
      in_t = torch.Tensor(Cells * CellVars),
      out_t = torch.Tensor(OutVars),
      cr = nn.MSECriterion()
   }
   L.net:zeroGradParameters()

   setmetatable(L, mt)
   return L
end

M.OutVars = OutVars
return M
