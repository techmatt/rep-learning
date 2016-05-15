require 'torch'
require 'nn'

local M = {}


function M.parsefile(p)
  local f = assert(io.open(p.file,'r'))
  assert(p.file and p.Kdim and p.numElems and p.T)
  M.rawdata = torch.Tensor(p.numElems,p.T,p.Kdim)
  
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

  for i=1,p.numElems do
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

--TODO 
function M.getData(p)
  assert(M.rawdata)
  assert(p.Tp and p.Tn and p.Kdim)
  numElems = M.rawdata:size()[1]
  local data = {}
  data.XPacked = torch.Tensor(numElems,p.Tp,p.Tp,p.Kdim)
  data.X = torch.Tensor(numElems,p.Tp,p.Kdim)
  data.Y = torch.Tensor(numElems,p.Tn,p.Kdim)
  for t=1,p.Tp do
    data.XPacked[{{},t}] = M.rawdata[{{},{t,t+p.Tp-1},{}}]:clone()
    data.X[{{},t}] = M.rawdata[{{},t,{}}]:squeeze()
  end
  for t=p.Tp+1,p.Tp+p.Tn do
    data.Y[{{},t-p.Tp}] = M.rawdata[{{},{t},{}}]:squeeze()
  end
  return data
end

return M
