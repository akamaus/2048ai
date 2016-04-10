package.path = package.path .. ';./?.lua'

local GP = require 'gnuplot'
local P = require 'plotter'

local Board = require 'board'

-- folds table t using function f and start state s0
function table.fold(t, f, s0)
   local s = s0 or 0
   for i,v in ipairs(t) do
      s = f(s, v)
   end
   return s
end

-- find max_ind and max of function applied to k1..k2
function find_max(k1, k2, f)
   local best = nil
   local best_k = nil

   for k=k1,k2 do
      local v = f(k)
      if (best_k == nil or v > best) then
         best_k, best = k, v
      end
   end
   return best_k, best
end

function sleep(s)
  local ntime = os.time() + s
  repeat until os.time() > ntime
end

-- Generate single episode on board b using policy
function gen_episode(b, policy)
   local states = {}
   repeat
--      print("*************")
      b:RandomGen()
--      b:Print()
--      print("")
      local move = policy(b)
      b:Move(move)
      states[#states + 1] = b:Compress()
      local finish = b:IsTerminal()
   until finish
   return states
end

-- Generate a bunch of episodes and draw length histogram
function sample_episodes(N, policy)
   local tab = {}

   for i=1,N do
      local b = Board.new()
      local states = gen_episode(b, policy)
      tab[#tab + 1] = #states
   end

   print("avg", table.fold(tab, function(a,b) return a+b end) / N)

   GP.hist(torch.Tensor(tab))
end

-- Neural network policy
local Cells = Board.S * Board.S -- num of elems on board
local CellVars = 16 -- num of elem variants

local NN = require 'nn'
-- build Torch-NN network
function build_net(w)
   local net = NN.Sequential()
   net:add(nn.Linear(Cells * CellVars,1))
--   net:add(nn.Tanh())
--   net:add(nn.Linear(w,1))

   return net
end

-- write board state into Tensor
function encode_state(t, st)
   t:fill(0)
   for i=1,Cells do
      for j=1,st:at(i) do
         t[(i-1) * CellVars + j] = 1
      end
   end
end

function draw_state(t, st)
   encode_state(t, st)
   local img = t:clone()
   GP.imagesc(img:resize(Cells, CellVars))
end

local in_t = torch.Tensor(Cells * CellVars)
local out_t = torch.Tensor(1)

-- policy based on NN decision
function nn_policy(net, b)
   local b_tst = Board.new()
   function try_move(m)
      Board.copy(b_tst, b)
      b_tst:Move(m)
      encode_state(in_t, b_tst:Compress())
      local val = net:forward(in_t)
      return val[1]
   end

   best_move, best_val = find_max(1,4, try_move)
   return best_move
end

local F_print = 100
local F_draw = 500
local F_est = 2000

function learn_policy(N, net)
   local log_val = torch.Tensor(N):zero()
   local log_err = torch.Tensor(N):zero()

   local avg_val = 0
   local avg_err = 0

   local Tau = 0.99

   local cr = nn.MSECriterion()

   function nn_pol(b)
      return nn_policy(net, b)
   end

   for i=1,N do
      local b = Board.new()
      local states = gen_episode(b, nn_pol )

      local preds = {}

      net:zeroGradParameters()
      local mse = 0
      for si,st in ipairs(states) do
         out_t[1] = #states - si
         encode_state(in_t, st, si)
         local v = net:forward(in_t)
         net:backward(in_t, cr:backward(v, out_t))
         mse = mse + (v[1] - out_t[1])*(v[1] - out_t[1])
         preds[#preds + 1] = v[1]

         if i % F_print == 0 then
            io.write(string.format("%0.2f ", v[1]))
         end

         if (si == 100) then
            break
         end
--         print("best_move,best_val", best_move, best_val)
      end
      net:updateParameters(0.0001)

      local err = mse / #states

      avg_val = avg_val * Tau + (1-Tau) * #states
      avg_err = avg_err * Tau + (1-Tau) * err

      log_val[i] = avg_val
      log_err[i] = avg_err

      if i % F_print == 0 then
         print()
         print("K", i, "avg val", avg_val, "val", #states, "err", mse / #states)
      end

      if i % F_draw == 0 then
         P.plot_table(preds)
      end

      if i % F_est == 0 then
         sample_episodes(1000, nn_pol)
      end

--      io.read "*line"
   end

   P.plot_tensors(log_err, log_val)
   --   print("avg", table.fold(tab, function(a,b) return a+b end) / N)
   --   GP.hist(torch.Tensor(tab))
end

-- interactive play, draws encoding
function interactive()
   local b = Board.new()
   local stop = false

   repeat
      b:RandomGen()
      b:Print()
      draw_state(in_t, b:Compress())
      local move = tonumber(io.read("*line"))
      if (move) then
         b:Move(move)
      end
   until stop
end

--interactive()

local N = 100000
local net = build_net(10)

learn_policy(N, net)



--GP.imagesc(t)
--sample_episodes(N, function() return math.random(4) end)


