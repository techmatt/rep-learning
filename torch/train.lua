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

local p = {}
p.Tp = 3
p.Tn = 3
p.Kdim = 4
p.Hdim = 20
p.initLayers = {20,20,p.Hdim}
p.recSharedLayers = {36,20}
p.recKLayers = {20,p.Kdim}
p.recHLayers = {20,p.Hdim}

print(p.recKLayers)

network.printNet(p)

getData.parsefile{
  file = "../datasets/simpleBall2k.txt",
  Kdim=p.Kdim,
  T=2*p.Tp+p.Tn,
  numElems=2000
}

data = getData.getData{
  Tp=p.Tp,
  Tn=p.Tn,
  Kdim=p.Kdim
}
--print(data.XPacked:size())
--print(data.X:size())
--print(data.Y:size())

rnn = network.createNet(p)

local function SGD(p)
  assert(p.data and p.mod and p.Tn and p.lr and p.batchSize and p.epochs)
  trainSize = p.data.X:size()[1]
  print(trainSize)
  assert(trainSize%p.batchSize==0)
  perEpoch = trainSize/p.batchSize
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
      dloss[p.Tn+1] = out[p.Tn+1]
      mod:zeroGradParameters()
      mod:backward({p.data.XPacked[bSel],p.data.X[bSel]},dloss)
      mod:updateParameters(lr)
      if(i%16==0) then print(loss) end
    end
    if(e%3==0) then
      p.lr = p.lr/3
    end
  end
end

SGD{
  data=data,
  mod=rnn,
  Tn=p.Tn,
  lr=lr,
  batchSize=batchSize,
  epochs=epochs
}

--print(mod.mods[5].weight == mod.mods[7].weight)
--print(mod.mods[1].mods[2].mods[2].weight == mod.mods[1].mods[4].mods[2].weight)
