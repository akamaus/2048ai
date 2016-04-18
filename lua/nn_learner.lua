
local GP = require 'gnuplot'

require 'helpers'
local Board = require 'board'
local NN = require 'nn'

local M = {}
-- Neural network policy
local Cells = Board.S * Board.S -- num of elems on board
local CellVars = 16 -- num of elem variants
local OutVars = 1

-- build Torch-NN network
local function build_net(name, w)
   name = name or ""
   local net = NN.Sequential()
   local first_layer = Cells * CellVars
   local last_layer = first_layer
   local formula = tostring(last_layer)
   for i, sz in pairs(w) do
      net:add(nn.Linear(last_layer,sz))
      net:add(nn.ReLU())
      last_layer = sz
      formula = formula .. ' ' .. tostring(sz)
   end
   net:add(nn.Linear(last_layer,OutVars))
   formula = formula .. ' ' .. OutVars

   net.name = string.format("%s_layers%s", name, formula)

   return net
end

-- build Torch-NN network
local function build_conv_net(name, ofeatures, k_size, w)
   name = name or ""
   local net = NN.Sequential()
   net:add(nn.SpatialConvolution(CellVars, ofeatures, k_size,k_size))
   local first_layer = ofeatures * (Board.S - 2) * (Board.S - 2)  -- Cells * CellVars
   local last_layer = first_layer
   local formula = tostring(last_layer)
   print("fl", Board.S)

   net:add(nn.View(first_layer))

   for i, sz in pairs(w) do
      net:add(nn.Linear(last_layer,sz))
      net:add(nn.ReLU())
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
   for i=1,Board.S do
      for j = 1,Board.S do
         local k = st:at2d(i,j) + 1
         t[k][i][j] = 1
      end
   end
end

-- set output neuron
local function encode_output(t, val)
   t[1] = val
end

-- get most active output neuron
local function decode_output(t)
   return t[1]
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
         L.cr:forward(pred, L.out_t)
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
      net = build_conv_net(name, 10, 3, w),
      in_t = torch.Tensor(CellVars, Board.S, Board.S),
      out_t = torch.Tensor(OutVars),
      cr = nn.MSECriterion()
   }
   L.net:zeroGradParameters()

   setmetatable(L, mt)
   return L
end

M.OutVars = OutVars
return M
