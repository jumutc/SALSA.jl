#
# Software Lab for Advanced Machine Learning with Stochastic Algorithms
# Copyright (c) 2015 Vilen Jumutc, KU Leuven, ESAT-STADIUS
# License & help @ https://github.com/jumutc/SALSA.jl
# Documentation @ http://salsajl.readthedocs.org
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#

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

function reduce_sparsevec{T<:AbstractSparseMatrix}(sm::T,idx)
  nzval  = sm.nzval[idx]
  rowval = sm.rowval[idx]
  colptr = [1,length(rowval)+1]
  SparseMatrixCSC(sm.m,sm.n,colptr,rowval,nzval)
end

function reduce_sparsevec{T<:AbstractSparseVector}(sm::T,idx)
  nzval = sm.nzval[idx]
  nzind = sm.nzind[idx]
  SparseVector(sm.n,nzind,nzval)
end


make_float(x) = typeof(x) <: Number ? x : parse(Float64,x)
make_int(x) = typeof(x) <: Number ? round(Int,x) : parse(Int,x)
