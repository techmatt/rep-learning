require 'torch'
--require 'nn'
require 'nngraph'
--package.path = package.path .. ";/home/rdaly525/torch-rnn/?.lua"


local M = {}

function M.createNet(p)
  assert(p.batch and p.Tp and p.Kdim and p.Hdim and p.initLayers and p.Tn)
  
  --Save in namespace
  local ns = {}

  --Set the inputs
  ns.XPacked = nn.Identity()()
  ns.X = nn.Identity()()
  
  --Keep array of Xs and Hs
  --Hstate = {[hdim],[hdim],[hdim]...}
  --Xinit = {[kdim],[kdim],[kdim]...}
  ns.Xstate = {}
  ns.Hstate = {}
  
  --Initialize Xstate and Hstate
  for t=1,p.Tp do
    ns.Xstate[t] = nn.SelectTable(t)(ns.X) 
    ns.Hstate[t] = nn.View(p.batch,p.Tp*p.Kdim)(nn.SelectTable(t)(ns.XPacked))
    local prev = p.Tp*p.Kdim
    for _,h in ipairs(p.initLayers) do
      ns.Hstate[t] = nn.Tanh()(nn.Linear(prev,h)(ns.Hstate[t]))
      prev = h
    end
  end

  
  stateLen = p.Tp*(p.Hdim+p.Kdim)
  ns.Xout = {}
  
  --
  for t=1,p.Tn do
    --Create State Tensor
    Xsel = nn.JoinTable(2)(nn.NarrowTable(t,t+p.Tp)(ns.Xstate))
    Hsel = nn.JoinTable(2)(nn.NarrowTable(t,t+p.Tp)(ns.Hstate))
    state = nn.JoinTable(2)({Hsel,Xsel})
    
    -- do the reccurance pass
    ns.Xstate[t+p.Tp] = nn.Linear(stateLen,p.Kdim)(state)
    ns.Hstate[t+p.Tp] = nn.Tanh()(nn.Linear(stateLen,p.Hdim)(state))
    
    --Add output
    ns.Xout[t] = ns.Xstate[p.Tp+t]
  end
  
  --Hack. Add last Hstate as output
  ns.Xout[p.Tn+1] = ns.Hstate[p.Tn+p.Tp]

  --Wrap inputs and outputs in nngraph's gModule
  mod = nn.gModule({ns.XPacked,ns.X},ns.Xout)
  

  --Share the weights
  ns.params, ns.gradParams = mod:parameters()
  print("INIT")
  print(ns.params)
  
  local function funprint(a,b)
    print(tostring(a).."<--"..tostring(b))
  end
  
  -- Share the parameters for the initial
  nI = #p.initLayers
  for t=1,p.Tp-1 do
    for i=1,2*nI do
      funprint(t*2*nI+i,i)
      --Share Weights and Biases
      ns.params[t*2*nI+i]:set(ns.params[i])
      ns.gradParams[t*2*nI+i]:set(ns.gradParams[i])
    end
  end
  
  --Check the params are shared 
  --Use this we use the module:share() function
  for t=1,p.Tp-1 do
    for i=1,2*nI do
      --funprint(t*2*nI+i,i)
      assert(torch.all(torch.eq(ns.params[t*2*nI+i],ns.params[i])))
      assert(torch.all(torch.eq(ns.gradParams[t*2*nI+i],ns.gradParams[i])))
    end
  end

  -- Share the parameters for the reccurance
  recurOffset = p.Tp*nI*2
  for t=1,p.Tn-1 do
    for i=1,4 do
      funprint(recurOffset+t*4+i,recurOffset+i)
      ns.params[recurOffset+t*4+i]:set(ns.params[recurOffset+i])
      ns.gradParams[recurOffset+t*4+i]:set(ns.gradParams[recurOffset+i])
    end
  end
  
  --Check the recur params are shared
  for t=1,p.Tn-1 do
    for i=1,4 do
      --funprint(recurOffset+t*4+i,recurOffset+i)
      assert(torch.all(torch.eq(ns.params[recurOffset+t*4+i],ns.params[recurOffset+i])))
      assert(torch.all(torch.eq(ns.gradParams[recurOffset+t*4+i],ns.gradParams[recurOffset+i])))
    end
  end

 
  
  print("AFTER")
  print(ns.params)
  mod.ns = ns
  return mod
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

