package.path = package.path .. ';./?.lua'

local GP = require 'gnuplot'

local Board = require 'board'

function table.fold(t, f, s0)
   local s = s0 or 0
   for i,v in ipairs(t) do
      s = f(s, v)
   end
   return s
end

-- Generate single episode using policy
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

--local N = 10000
--sample_episodes(N, function() return math.random(4) end)

-- Neural network policy
local Cells = Board.S * Board.S -- num of elems on board
local CellVars = 16 -- num of elem variants

function build_net(w)
   local net = nn.Sequential()
   net:add(nn.Linear(Cells * CellVars,1))
   return net
end

function encode_state(t, st)
   t:fill(0)
   for i=1,Cells do
      for j=1,st:at(i) do
         t[i][j] = 1
      end
   end
end



local b = Board.new()
b:RandomGen()
b:RandomGen()
b:Move(2)
b:Move(1)

t = torch.Tensor(Cells, CellVars)
encode_state(t, b:Compress())

GP.imagesc(t)
