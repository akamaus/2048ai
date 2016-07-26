
local GP = require 'gnuplot'

require 'helpers'
local Board = require 'board'
local NN = require 'nn'

local M = {}
-- Neural network policy
local Cells = Board.S * Board.S -- num of elems on board
local CellVars = 16 -- num of elem variants
local OutVars = 4

local alpha = 0.9 -- backup rate

-- build Torch-NN network, Estimates Q(s) = [a1,a2,a3,a4]
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
   t[1] = val
end

-- get most active output neuron and its activation level
local function decode_output(t)
  local q_ten,a_ten = torch.max(t,1)
  return a_ten[1],q_ten[1]
end

local function draw_input(t, st)
   local img = t:clone()
   GP.imagesc(img:resize(Cells, CellVars))
end

local function learn_minibatch(self, batch)
  local err = 0
  for i=1, #batch do
    local sample = batch[i]
    encode_input(self.in_t, sample.s0)
    local preds = self.net:forward(self.in_t)
    for i=1, OutVars do
      self.out_t[i] = preds[i]
    end
    local _, v1 = self:best_move(sample.s1)

    self.out_t[sample.action] = alpha * self.out_t[sample.action] + (1 - alpha) * (sample.reward + v1)

    self.cr:forward(preds, self.out_t)
    self.net:backward(self.in_t, self.cr:backward(preds, self.out_t))
    -- TODO err
  end
  return err
end

local mt = {
   __index = {
      -- returns estimated value of a position
      best_move = function(L, st)
         encode_input(L.in_t, st)
         local pred = L.net:forward(L.in_t)
         return decode_output(pred)
      end,
      -- learns true value and returns error
      learn = learn_minibatch,
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
      cr = nn.MSECriterion(),
      type = 'q-learner'
   }
   L.net:zeroGradParameters()

   setmetatable(L, mt)
   return L
end

M.OutVars = OutVars
return M
