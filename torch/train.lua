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

local lr = 0.001
local epochs = 10

local batchSize = 5
local Tp = 3
local T = 12
local Tn = 5
local Kdim = 4
local Hdim = 11
local initLayers = {12,12,12,Hdim}
--local recLayers = {13,15,18,Hdim}


getData.parsefile{
  file = "../datasets/simpleBall2k.txt",
  Kdim=Kdim,
  T=T,
  numElems=2000
}

data = getData.getData{
  Tp=Tp,
  Tn=Tn,
  Kdim=Kdim
}
print(data.XPacked:size())
print(data.X:size())
print(data.Y:size())

mod = network.createNet{
  Tp=Tp,
  Kdim=Kdim,
  Hdim=Hdim,
  initLayers=initLayers,
  Tn=Tn
}

local function SGD(p)
  assert(p.data and p.mod and p.Tn and p.lr and p.batchSize and p.epochs)
  trainSize = p.data.X:size()[1]
  print(trainSize)
  assert(trainSize%p.batchSize==0)
  perEpoch = trainSize/p.batchSize
  local lr = p.lr
  for e=1,p.epochs do
    print("----------EPOCH "..tostring(e))
    for i=0,perEpoch-1 do
      --Simple batchSelector
      bSel = {{i*p.batchSize+1,(i+1)*p.batchSize}}
      out = mod:forward({p.data.XPacked[bSel],p.data.X[bSel]})
      
      --Use MSE
      criterion = nn.MSECriterion()
      
      --calculate loss and backprop MSE for each output
      loss=0
      dloss = {}
      for t=1,p.Tn do
        local sel = {{i*p.batchSize+1,(i+1)*p.batchSize},t}
        loss = loss + criterion:forward(out[t],p.data.Y[sel])
        dloss[t] = criterion:backward(out[t],p.data.Y[sel])
      end
      dloss[Tn+1] = out[Tn+1]
      mod:zeroGradParameters()
      mod:backward({p.data.XPacked[bSel],p.data.X[bSel]},dloss)
      mod:updateParameters(lr)
      if(i%16==0) then print(loss) end
    end
    if(e%2==0) then
      lr = lr/3
    end
  end
end

SGD{
  data=data,
  mod=mod,
  lr=lr,
  Tn=Tn,
  batchSize=batchSize,
  epochs=epochs
}

--print(mod.mods[5].weight == mod.mods[7].weight)
--print(mod.mods[1].mods[2].mods[2].weight == mod.mods[1].mods[4].mods[2].weight)
