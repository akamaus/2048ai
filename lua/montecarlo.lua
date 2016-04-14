package.path = package.path .. ';./?.lua'

local GP = require 'gnuplot'
local P = require 'plotter'

local Board = require 'board'

require 'helpers'

--local TL = require 'table_learner'
local NN = require 'nn_learner'


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


-- policy based on NN decision
function eps_greedy_policy(learner, eps, b)
   if math.random() < eps then
      return math.random(1,4),0
   end

   local b_tst = Board.new()
   function try_move(m)
      Board.copy(b_tst, b)
      b_tst:Move(m)
      return learner:est_value(b_tst:Compress())
   end

   best_move, best_val = find_max(1,4, try_move)
   return best_move
end

local F_print = 100
local F_draw = 100
local F_est = 10000
local F_save = 100000

local LRate = 0.01
local Eps = 0.001

function learn_policy(container)
   local learner = container.learner
   local avg_val = container.avg_val

   local Tau = 0.99

   function pol(b)
      return eps_greedy_policy(learner, Eps, b)
   end

   local preds_ten = torch.Tensor(1000, NN.OutVars)
   local preds = {}

   while container.i <= container.N do
      local i = container.i

      local b = Board.new()
      local states = gen_episode(b, pol)

      local mse = 0
      for si,st in ipairs(states) do
         local val = #states - si
         local err, pred = learner:learn(st, val)
         mse = mse + err*err
         every(F_draw, i, function()
                  preds[#preds + 1] = val + err
                  preds_ten[si] = pred
         end)
      end

      avg_val = avg_val * Tau + (1-Tau) * #states
      learner:apply(LRate)

      -- store learning curve point
      every(F_print / 10, i, function() container.log_val[i] = avg_val end)

      local err = mse / #states

      -- print progress
      every(F_print, i, function() print("K", i, "NS", learner.num_states, "avg val", avg_val, "val", #states, "err", mse / #states) end )

      every(F_draw,i,
            function()
               P.with_multiplot(1,2,
                                function()
                                   local pr  = preds_ten:narrow(1, 1, #states):transpose(1,2)
                                   GP.imagesc(pr)
                                   P.plot_table(preds)
                                   preds = {}
                                end
               )
            end
      )

      every(F_est,i, function() sample_episodes(1000, pol) end)

      container.i = container.i + 1

      every(F_save, i,
            function()
               local path = string.format("checkpoints/%s_iter_%d_avg_%0.2f.sav", learner.name, i, avg_val)
               container.avg_val = avg_val
               torch.save(path, container)
               print("saved " .. path)
            end
      )
   end

   P.plot_tensors(container.log_val)
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

function build_container(N, learner)
   local c = {
      N = N,
      i = 1,
      learner = learner,
      log_val = {},
      avg_val = 0
   }

   return c
end

--
-- Analysis
--

-- draw layer as it evolves through success checkpoints
function draw_layer_evolution(checkpoints)
   GP.raw('set cbrange [-40:40]')
   GP.raw('set palette defined (-50 "blue", 0 "white", 50 "red")')

   for i,f in ipairs(checkpoints) do
      cont = torch.load(f)
      GP.imagesc(cont.net.modules[1].weight:resize(Cells,CellVars), 'color')
      sleep(1)
   end
--   io.read("*l")
end

-- draw performance of multiple checkpoints on a single game
function evaluate_checkpoints(files, seed)
   local vals1 = {}
   for i,f in ipairs(files) do
      cont = torch.load(f)
      b = Board.new()
      Board.srand(seed)
      local states1 = gen_episode(b, function(b)
                              return nn_policy(cont.net, b, 0)
                              end)
      vals1[#vals1 + 1] = #states1
   end
   P.plot_table(vals1)
end

-- learn a number of networks
function learn_series(N, seed_num)
   for i=1, seed_num do
      torch.manualSeed(1)
      math.randomseed(0)
      Board.srand(i)
      cont = build_container(N, 100, "seed" .. i)
      learn_policy(cont)
   end
end

function draw_layer_variants(checkpoints)
   GP.raw('set cbrange [-40:40]')
   GP.raw('set palette defined (-50 "blue", 0 "white", 50 "red")')
   GP.raw('unset xtics')

   P.with_multiplot(3,3,
                    function()
                       for i,f in ipairs(checkpoints) do
                          cont = torch.load(f)
                          GP.imagesc(cont.net.modules[1].weight:resize(Cells,CellVars), 'color')
                       end
                    end
   )
end

local cont
-- if #arg == 1 then
--    cont = torch.load(arg[1])
--    cont.learner = TL.build_table_learner(nil, cont.learner)
-- else
--    cont = build_container(tonumber(arg[2]), TL.build_table_learner(arg[1]))
-- end

if #arg == 1 then
   cont = torch.load(arg[1])
   cont.learner = NN.build_nn_learner(nil, nil, cont.learner)
else
   cont = build_container(tonumber(arg[2]), NN.build_nn_learner(arg[1], {}))
end

learn_policy(cont)

--learn_series(tonumber(arg[1]), tonumber(arg[2]))

--cont = torch.load(arg[1])
--cont.learner = build_nn_learner(cont.learner)
--learn_policy(cont)

--draw_layer_evolution(arg)
--draw_layer_variants(arg)

--local N = 10000

--

--draw_layers(arg)



--interactive()
--local net = build_net(100)
--sample_episodes(N, function() return math.random(4) end)

