require 'torch'
--require 'nn'
require 'nngraph'
--package.path = package.path .. ";/home/rdaly525/torch-rnn/?.lua"



-- Ask Zack how to do to find this
--gModule = torch.getmetatable('nn.gModule')
-- this is redefinition of the share() method for gModule from nnGraph
--function gModule:share(gModuleToShare, ...)
--  for indexNode, node in ipairs(self.forwardnodes) do
--    if node.data.module then
--      node.data.module:share(gModuleToShare.forwardnodes[indexNode].data.module, ...)
--    end
--  end
--end


-- TODO, intigrate this into nngraph like above
function share(a,b, ...)
  if a.data.module and b.data.module then
    a.data.module:share(b.data.module, ...)
  else 
    print("WARNING: Did not share module")
    os.exit(1)
  end
end

function shareLinear(a,b)
  share(a,b,'weight','gradWeight','bias','gradBias')
end


local M = {}

function M.createNet(p)
  assert(p.Tp and p.Kdim and p.Hdim and p.initLayers and p.Tn and
    p.recSharedLayers and p.recKLayers and p.recHLayers)


  --Save in namespace
  local ns = {}

  --Set the inputs
  ns.KPacked = nn.Identity()()
  ns.K = nn.Identity()()
  
  --Keep array of Ks and Hs
  --Hstate = {[hdim],[hdim],[hdim]...}
  --Kstate = {[kdim],[kdim],[kdim]...}
  ns.Kstate = {}
  ns.Hstate = {}
  
  --Initialize Kstate and Hstate
  ns.initLinear = {}
  for t=1,p.Tp do
    ns.Kstate[t] = nn.Select(2,t)(ns.K)
    local prevDim = p.Tp*p.Kdim
    local prevLayer = nn.View(p.Tp*p.Kdim)(nn.Select(2,t)(ns.KPacked))
    ns.initLinear[t] = {}
    for i,hDim in ipairs(p.initLayers) do
      ns.initLinear[t][i] = nn.Linear(prevDim,hDim)(prevLayer)
      shareLinear(ns.initLinear[t][i],ns.initLinear[1][i])
      prevLayer = nn.Tanh()(ns.initLinear[t][i])
      prevDim = hDim
    end
    ns.Hstate[t] = prevLayer
  end
  
  stateLen = p.Tp*(p.Hdim+p.Kdim)
  print("StateLen: "..tostring(stateLen))
  ns.Kout = {}
  --
  --8x48 -> (48,48) -> 8x48
  ns.recShared={}
  ns.recK = {}
  ns.recH = {}
  for t=1,p.Tn do
    --Create State Tensor
    Ksel = nn.JoinTable(2)(nn.NarrowTable(t,t+p.Tp)(ns.Kstate))
    Hsel = nn.JoinTable(2)(nn.NarrowTable(t,t+p.Tp)(ns.Hstate))
    state = nn.JoinTable(2)({Hsel,Ksel})
    
    --Do the reccurance pass
    --Shared first
    ns.recShared[t]={}
    local prevDim = stateLen
    local prevLayer = state
    for i,hDim in ipairs(p.recSharedLayers) do
      ns.recShared[t][i] = nn.Linear(prevDim,hDim)(prevLayer)
      shareLinear(ns.recShared[t][i],ns.recShared[1][i])
      prevLayer = nn.Tanh()(ns.recShared[t][i])
      prevDim = hDim
    end
    local sharedLayerDim = prevDim
    local sharedLayer= prevLayer
    
    --do layers for K
    ns.recK[t] = {}
    for i,hDim in ipairs(p.recKLayers) do
      ns.recK[t][i] = nn.Linear(prevDim,hDim)(prevLayer)
      shareLinear(ns.recK[t][i],ns.recK[1][i])
      prevDim = hDim
      if (i==#p.recKLayers) then
        prevLayer = ns.recK[t][i]
      else
        prevLayer = nn.Tanh()(ns.recK[t][i])
      end
    end
    ns.Kstate[t+p.Tp] = prevLayer
    --set output
    ns.Kout[t] = prevLayer

    --do layers for H
    ns.recH[t] = {}
    prevDim = sharedLayerDim
    prevLayer = sharedLayer
    for i,hDim in ipairs(p.recHLayers) do
      ns.recH[t][i] = nn.Linear(prevDim,hDim)(prevLayer)
      shareLinear(ns.recH[t][i],ns.recH[1][i])
      prevDim = hDim
      prevLayer = nn.Tanh()(ns.recH[t][i])
    end
    ns.Hstate[t+p.Tp] = prevLayer
    
  end
  
  --Hack. Add last Hstate as output
  ns.Kout[p.Tn+1] = ns.Hstate[p.Tn+p.Tp]

  --Wrap inputs and outputs in nngraph's gModule
  mod = nn.gModule({ns.KPacked,ns.K},ns.Kout)
  

  --Share the weights
  ns.params, ns.gradParams = mod:parameters()
 
  --local function funprint(a,b)
  --  print(tostring(a).."<--"..tostring(b))
  --end
  
  -- Share the parameters for the initial
  --nI = #p.initLayers
  --for t=1,p.Tp-1 do
  --  for i=1,2*nI do
  --    --funprint(t*2*nI+i,i)
  --    --Share Weights and Biases
  --    ns.params[t*2*nI+i]:set(ns.params[i])
  --    ns.gradParams[t*2*nI+i]:set(ns.gradParams[i])
  --  end
  --end
  
  --Check the params are shared 
  --Use this we use the module:share() function
  --for t=1,p.Tp-1 do
  --  for i=1,2*nI do
  --    --funprint(t*2*nI+i,i)
  --    assert(torch.all(torch.eq(ns.params[t*2*nI+i],ns.params[i])))
  --    assert(torch.all(torch.eq(ns.gradParams[t*2*nI+i],ns.gradParams[i])))
  --  end
  --end

  -- Share the parameters for the reccurance
  --recurOffset = p.Tp*nI*2
  --for t=1,p.Tn-1 do
  --  for i=1,4 do
  --    --funprint(recurOffset+t*4+i,recurOffset+i)
  --    ns.params[recurOffset+t*4+i]:set(ns.params[recurOffset+i])
  --    ns.gradParams[recurOffset+t*4+i]:set(ns.gradParams[recurOffset+i])
  --  end
  --end
  
  --Check the recur params are shared
  --for t=1,p.Tn-1 do
  --  for i=1,4 do
  --    funprint(recurOffset+t*4+i,recurOffset+i)
  --    assert(torch.all(torch.eq(ns.params[recurOffset+t*4+i],ns.params[recurOffset+i])))
  --    assert(torch.all(torch.eq(ns.gradParams[recurOffset+t*4+i],ns.gradParams[recurOffset+i])))
  --  end
  --end

  mod.ns = ns
  return mod
end

function M.printNet(p)
  assert(p.Tp and p.Kdim and p.Hdim and p.initLayers and p.Tn and
    p.recSharedLayers and p.recKLayers and p.recHLayers)
  
  local initLen = p.Tp*p.Kdim
  local stateLen = p.Tp*(p.Hdim+p.Kdim)
  print("Network!")
  io.write("Init: "..initLen)
  for i,h in ipairs(p.initLayers) do
    io.write("->"..(h))
  end
  io.write("\nrecShared: "..stateLen)
  for i,h in ipairs(p.recSharedLayers) do
    io.write("->"..(h))
  end
  sharedLen = p.recSharedLayers[#p.recSharedLayers]
  io.write("\nrecK: "..sharedLen)
  for i,h in ipairs(p.recKLayers) do
    io.write("->"..(h))
  end
  io.write("\nrecH: "..sharedLen)
  for i,h in ipairs(p.recHLayers) do
    io.write("->"..(h))
  end
  print("\n")
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

