function make_sparse(tuples;sizes=(),delim="")
  (N,d) = size(tuples)
  index = delim == "" ? (1:2:d-1) : (1:d)

  I = @compat Vector{Int}()
  J = @compat Vector{Int}()
  V = @compat Vector{Float64}()

  # find all the indices and values
  for i in 1:N
    for j in index
      if tuples[i,j] != ""
        if delim == ""
          push!(V,make_float(tuples[i,j+1]))
          push!(J,make_int(tuples[i,j]))
          push!(I,i)
        else
          tuple = split(tuples[i,j],delim)
          push!(V,make_float(tuple[2]))
          push!(J,make_int(tuple[1]))
          push!(I,i)
        end  
      end
    end
  end

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

make_float(x) = typeof(x) <: Number ? x : parse(Float64,x)
make_int(x) = typeof(x) <: Number ? round(Int,x) : parse(Int,x)