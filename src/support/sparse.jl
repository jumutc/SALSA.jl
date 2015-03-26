function make_sparse(tuples;sizes=(),delim="")
  (N,d) = size(tuples); cnt = 1
  I::Vector{Int} = zeros(1)
  J::Vector{Int} = zeros(1)
  V::Vector{Float64} = zeros(1)
  index = delim == "" ? (1:2:d-1) : (1:d) 

  # find all the indices and values
  for i in 1:N
    for j in index
      if tuples[i,j] != ""
        if delim == ""
          V[cnt] = tuples[i,j+1]
          J[cnt] = tuples[i,j]
          I[cnt] = i
        else
          tuple = split(tuples[i,j],delim)
          V[cnt] = float(tuple[2])
          J[cnt] = int(tuple[1])
          I[cnt] = i
        end
        resize!(I,cnt+1)
        resize!(J,cnt+1)
        resize!(V,cnt+1)
        cnt+=1
      end
    end
  end
  #reshape it to normnal
  resize!(I,cnt-1)
  resize!(J,cnt-1)
  resize!(V,cnt-1)

  # check for available sizes
  if ~isempty(sizes) 
    sparse(I,J,V,sizes[1],sizes[2])
  else 
    sparse(I,J,V)
  end
end

function reduce_sparsevec(sm::SparseMatrixCSC,idx)
  nzval  = sm.nzval[idx]
  rowval = sm.rowval[idx]
  colptr = [1,length(rowval)+1]
  SparseMatrixCSC(sm.m,sm.n,colptr,rowval,nzval)
end