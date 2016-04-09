package.path = package.path .. ';./?.lua'

local Board = require 'board'

function table.fold(t, f, s0)
   local s = s0 or 0
   for i,v in ipairs(t) do
      s = f(s, v)
   end
   return s
end

function gen_episode(b, policy)
   local counter = 0
   repeat
--      print("*************")
      b:RandomGen()
--      b:Print()
--      print("")
      local move = policy(b)
      b:Move(move)
--      b:Print()
      local finish = b:IsTerminal()
      counter = counter + 1
   until finish
   return counter
end

function sample_episodes(N, policy)
   local tab = {}

   for i=1,N do
      local b = Board.new()
      local n = gen_episode(b, policy)
      tab[#tab + 1] = n
   end

   print("avg", table.fold(tab, function(a,b) return a+b end) / N)

   local GP = require 'gnuplot'
   GP.hist(torch.Tensor(tab))
end

local N = 10000

sample_episodes(N, function() return math.random(4) end)
