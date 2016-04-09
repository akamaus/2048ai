package.path = package.path .. ';./?.lua'

local Board = require 'board'

function gen_episode(b)
   local counter = 0
   repeat
--      print("*************")
      b:RandomGen()
--      b:Print()
--      print("")
      local move = math.random(4)
      b:Move(move)
--      b:Print()
      local finish = b:IsTerminal()
      counter = counter + 1
   until finish
   return counter
end


local tab = {}

for i=1,10000 do
   local b = Board.new()
   local n = gen_episode(b)
   tab[#tab + 1] = n
end

local GP = require 'gnuplot'
GP.hist(torch.Tensor(tab))

--P.plot_table(tab)

