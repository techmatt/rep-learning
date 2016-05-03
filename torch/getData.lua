require 'torch'
require 'nn'

local M = {}

local function mysplit(inputstr,sep)
  if sep==nil then
    sep=","
  end
  local t={}; i=1
  for str in string.gmatch(inputstr,"([^"..sep.."]+)") do
    t[i] = str
    i = i+1
  end
  return t
end

--takes in
--file
--batch
--T
--Kdim


function M.getRawData(p)
  local f = assert(io.open(p.file,'r'))
  assert(p.file and p.Kdim and p.batch and p.T)
  M.rawdata = torch.Tensor(p.batch,p.T,p.Kdim)

  for i=1,p.batch do
    local line = f:read()
    if line==nil then break end
    local arr = mysplit(line)
    for t=1,p.T do
      for j=1,4 do
        M.rawdata[i][t][j] = arr[(t-1)*4+j]
      end
    end
  end
  f.close()
  --return rawdata
end

--Fucking slicing a:b includes both a and b
--Need clone because the data needs to be contiguous
--

--X dimensions
--odd: X[2*Tp] = (Batch,Kdim) --contains the first Tp K values
--even: X[2*Tp] = (Batch,Tp,Kdim) --contains a window of Tp Values 
--Y dimensions
--TODO 
function M.getData(p)
  assert(M.rawdata)
  assert(p.Tp)
  local data = {}
  data.X = {}
  for t=1,2*p.Tp do
    if t%2==1 then
      data.X[t] = M.rawdata[{{},{(t-1)/2+1},{}}]:squeeze()
    else
      data.X[t] = M.rawdata[{{},{t/2,t/2+p.Tp-1},{}}]:clone()
    end

  end

  return data
end

return M
