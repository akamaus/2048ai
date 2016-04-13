local M = {}

local Board = require 'board'

local mt = {
   __index = {
      -- getter, from raw position(luanumber)
      get_val = function(L, u)
         return L.states[u] or 1000
      end,
      -- returns estimated value of a state
      est_value = function(L, st)
         return L:get_val(st:u64())
      end,
      -- learns true value and returns error
      learn = function(L, st, val)
         local s = st:u64()
         L.deltas[s] = val - L:est_value(st)
         return L.deltas[s]
      end,
      -- applies learned material
      apply = function(L, rate)
         for s,d in ipairs(L.deltas) do
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
      deltas = {}
   }

   setmetatable(L, mt)
   return L
end

return M

