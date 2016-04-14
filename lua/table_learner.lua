local M = {}

local Board = require 'board'

local mt = {
   __index = {
      -- getter, from raw position(luanumber)
      get_val = function(L, u)
         if L.states[u] == nil then
            L.states[u] = 40
            L.num_states = L.num_states + 1
         end
         return L.states[u]
      end,
      -- returns estimated value of a state
      est_value = function(L, st)
         return L:get_val(st:u64())
      end,
      -- learns true value and returns error
      learn = function(L, st, val)
         local s = st:u64()
         L.deltas[s] = val - L:est_value(st)
         return -L.deltas[s]
      end,
      -- applies learned material
      apply = function(L, rate)
         for s,d in pairs(L.deltas) do
            L.states[s] = L:get_val(s) + rate * d
            L.deltas[s] = nil
         end
      end
   }
}

function M.build_table_learner(name, L)
   local L = L or {
      name = name,
      states = {},
      num_states = 0,
      deltas = {}
   }

   setmetatable(L, mt)
   return L
end

return M

