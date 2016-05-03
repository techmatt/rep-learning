require 'torch'
require 'nn'
--package.path = package.path .. ";/home/rdaly525/torch-rnn/?.lua"
--require 'LSTM'

network = require 'network'
getData = require 'getData'


local batch = 5
local Tp = 3
local T = 12
local Tn = 5
local Kdim = 4
local Hdim = 10
local initLayers = {10,10,Hdim}

getData.getRawData{
  file = "../datasets/simpleBall2k.txt",
  Kdim=Kdim,
  T=T,
  batch=batch
}

data = getData.getData{
  Tp=Tp,
}
print(data.X)

module = network.createNet{
  batch=batch,
  Tp=Tp,
  Kdim=Kdim,
  Hdim=Hdim,
  initLayers=initLayers,
  Tn=5
}
print(module)
out = module:forward(data.X)
print(out:size())

--criterion = nn.MSECriterion()
--out = module:forward(input)
--loss = criterion:forward(out,Y)
--dloss = criterion:backward(out,Y)
--module:backward(input,dLoss)

