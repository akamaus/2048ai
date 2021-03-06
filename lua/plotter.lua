local M = {}

local gp = require 'gnuplot'

require 'helpers'

function M.with_multiplot(i,j, acts)
   gp.raw('set terminal wxt size 1200,600')
   gp.raw(string.format('set multiplot layout %d,%d', i,j))
   gp.raw('unset key')
   if type(acts) == 'table' then
      for _,a in pairs(acts) do
         a()
      end
   elseif type(acts) == 'function' then
      acts()
   else error "pass function table or single function"
   end

   gp.raw("unset multiplot")
end

local function plot_table(tab)
   local min, max, num_elems = indexes(tab)

   if type(tab[min]) == "number" then
      local ten = torch.Tensor(num_elems,2)
      local i = 1
      for x,y in pairs(tab) do
         ten[i][1] = x
         ten[i][2] = y
         i = i + 1
      end
      gp.plot(ten, '~')
   elseif type(tab[min]) == "table" then
      local min2, max2 = indexes(tab[min])

      local ten = torch.Tensor(max - min + 1, max2-min2+1)
      local i=1
      for k=min,max do
         local j = 1
         for l = min2,max2 do
            ten[i][j] = tab[k][l]
            j = j + 1
         end
         i = i + 1
      end
      gp.imagesc(ten)
   else error("unknown elt")
   end
end

function M.plot_table(tab1, tab2)
   if tab2 then
      M.with_multiplot(1,2, {function() plot_table(tab1) end, function() plot_table(tab2) end})
   else
      plot_table(tab1)
   end
end

function M.plot_tensors(t1, t2)
   if t2 then
      M.with_multiplot(1,2, {function() gp.plot(t1, '~') end, function() gp.plot(t2, '~') end})
   else
      gp.plot(t1, '~')
   end
end


function M.hist_table(tab)
   local min,max = indexes(tab)
   local t= torch.Tensor(max)

   for i,v in ipairs(tab) do
      t[i] = v
   end
   gp.hist(t)
end

return M
