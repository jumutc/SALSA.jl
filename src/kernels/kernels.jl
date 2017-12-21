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

abstract type Kernel end

kernel_matrix(k::Kernel, X::Matrix) = kernel_matrix(k, X, X)
kernel_matrix(k::Kernel, X::Array{Float64,1}) = kernel_matrix(k, X')
kernel_matrix(k::Kernel, X::SubArray) = kernel_matrix(k, X[:,:], X[:,:])

kernel_matrix(k::Kernel, X::Array{Float64,1}, Xn::Array{Float64,2}) = kernel_matrix(k, X', Xn)
kernel_matrix(k::Kernel, X::Array{Float64,2}, Xn::Array{Float64,1}) = kernel_matrix(k, X, Xn')
kernel_matrix(k::Kernel, X::Array{Float64,1}, Xn::Array{Float64,1}) = kernel_matrix(k, X', Xn')
kernel_matrix(k::Kernel, X::SubArray, Xn::SubArray) = kernel_matrix(k, X[:,:], Xn[:,:])
kernel_matrix(k::Kernel, X::SubArray, Xn) = kernel_matrix(k, X[:,:], Xn)
kernel_matrix(k::Kernel, X, Xn::SubArray) = kernel_matrix(k, X, Xn[:,:])

kernel_matrix(k::Kernel, X::DelimitedFile, Xn) = kernel_matrix(k, sub(X,:,:), Xn)
kernel_matrix(k::Kernel, X::DelimitedFile) = begin Xf = sub(X,:,:); kernel_matrix(k, Xf, Xf) end

include("rbf_kernel.jl")
include("polynomial_kernel.jl")
include("linear_kernel.jl")

kernel_from_parameters{T<:Kernel}(k::Type{T}, parameters) = k(parameters...)
kernel_from_data_model{T<:Kernel}(k::Type{T}, X) = isempty(fieldnames(k)) ? k() : k(rand(length(fieldnames(k)))...)
