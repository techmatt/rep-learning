require 'torch'
require 'nn'
require 'nngraph'
--package.path = package.path .. ";/home/rdaly525/torch-rnn/?.lua"
--require 'LSTM'


--K_1,K_2,..,K_<Tp>,..,K_<Tp+Tn>


--TODO 
--Need to add recurrent part of network. How to actually share weights?
--Need to verify that the initFC is actually shared weights (it probably is not currently)
--Neet to make sure that the c

local M = {}

function M.createNet(p)
  assert(p.batch and p.Tp and p.Kdim and p.Hdim and p.initLayers and p.Tn)

  initNN = nn.Sequential()

  initFC = nn.Sequential()
  initFC:add(nn.View(p.batch,p.Tp*p.Kdim))
  local prev = p.Tp*p.Kdim
  for _,h in pairs(p.initLayers) do
    initFC:add(nn.Linear(prev,h))
    initFC:add(nn.ReLU())
    prev = h
  end

  initP = nn.ParallelTable()
  for t=1,p.Tp do
    initP:add(nn.Identity())
    initP:add(initFC)
  end
  initNN:add(initP)
  initNN:add(nn.JoinTable(2))
  return initNN

end

return M


----input initialization, get hidden state
--K[1:T] -> H[T]
--K[2:T+1] -> H[T+1]
--...
--K[T:2*T] -> H[2*T]
--
----Main Recurrance
----Init:
--  recIn = {H[2*T],K[2*T],..,H[T],K[T]} 
----
--  recOut = {H[X+1],K[X+1]} = FC(recIn)
--  recIn = {recOut,recIn[1:2*T-2]}

