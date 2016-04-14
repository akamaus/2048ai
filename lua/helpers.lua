
-- sleeps s seconds
function sleep(s)
  local ntime = os.time() + s
  repeat until os.time() > ntime
end

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

function every(K, i, f)
   if math.floor(i) % K == 0 then
      f()
   end
end
