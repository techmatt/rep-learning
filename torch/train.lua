require 'torch'
require 'nn'
--package.path = package.path .. ";/home/rdaly525/torch-rnn/?.lua"
--require 'LSTM'

network = require 'network'
getData = require 'dataset'

local function ptable(T)
  
  for k,v in T do
    print(k)
    print(v)
    --print("("..tostring(k) .. ","..tostring(v)..")")
  end
end

local lr = .01

local batch = 5
local Tp = 3
local T = 12
local Tn = 5
local Kdim = 4
local Hdim = 10
local initLayers = {7,9, Hdim}
local recLayers = {13,15,18,Hdim}


getData.parsefile{
  file = "../datasets/simpleBall2k.txt",
  Kdim=Kdim,
  T=T,
  numElems=2000
}

data = getData.getData{
  Tp=Tp,
  Tn=Tn
}
print(data.XPacked)
print(data.X)
print(data.Y)
print(data.X[1]:size()[1])

mod = network.createNet{
  batch=batch,
  Tp=Tp,
  Kdim=Kdim,
  Hdim=Hdim,
  initLayers=initLayers,
  Tn=Tn
}

local function train(p)
  assert(p.data and p.mod and p.lr and p.batchSize and p.epochs)
  trainSize = p.data.X[1].size()[1]
  assert(p.trainSize%p.batchSize==0)
  perEpoch = p.trainSize/p.batchSize
  for e=1,p.epochs do
    for i=0,perEpoch do
      bSel = {{i*p.batchSize,(i+1)*p.batchSize}}
      out = mod:forward({data.XPacked[bSel],data.X[bSel]})
      print(out)
      criterion = nn.MSECriterion()
      loss=0
      dloss = {}
      for t=1,Tn do
        loss = loss + criterion:forward(out[t],data.Y[t])
        dloss[t] = criterion:backward(out[t],data.Y[t])
      end
      dloss[Tn+1] = out[Tn+1]
      mod:zeroGradParameters()
      mod:backward({data.Xpacked,data.X},dloss)
      mod:updateParameters(lr)
      print(loss)
    end
  end
end
--print(mod.mods[5].weight == mod.mods[7].weight)
--print(mod.mods[1].mods[2].mods[2].weight == mod.mods[1].mods[4].mods[2].weight)
