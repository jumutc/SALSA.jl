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

# fix for julia release where this function is absent, TODO: remove when we move to julia 0.4+
sub(a::SubArray, I::AbstractVector, ::Colon) = convert(Array, a[I,:])
sub(a::Matrix, I::AbstractVector, ::Colon) = a[I,:]
sub(a::Vector, I::AbstractVector, ::Colon) = a[I]
sub(a::Matrix, i::Int, ::Colon) = a[i,:]
sub(a::Matrix, ::Colon, ::Colon) = a
sub{T<:AbstractSparseArray}(a::T, ::Colon, ::Colon) = a
sub{T<:AbstractSparseArray}(a::T, I::AbstractVector, ::Colon) = sparse(a[I,:])
view{T<:AbstractSparseArray}(a::T, ::Colon, I::Int) = a[:,I]
dot{T<:AbstractSparseArray}(a::T, b::T) = sum(a.*b)
# append ones as a 'bias' term to the transposed subsample
append_ones(a::AbstractMatrix, num::Int) = [a'; ones(1,num)]
append_ones(a::AbstractVector, ::Int) = push!(a,1)
append_ones{T<:AbstractSparseMatrix}(a::T, num::Int) = vcat(a,sparse(ones(1,num)))
