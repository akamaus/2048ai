package.path = package.path .. ';./?.lua'

local Board = require 'board'

local b = Board.new()

b:RandomGen(b)
b:Print()
print(b:Compress())
print("down")
b:Move(Board.Down)
b:Print()
print(b:Compress())

