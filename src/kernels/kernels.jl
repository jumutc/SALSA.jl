export Kernel
abstract Kernel

kernel_matrix(k::Kernel, X::Matrix) = kernel_matrix(k, X, X)
kernel_matrix(k::Kernel, X::Array{Float64,1}) = kernel_matrix(k, X'')
kernel_matrix(k::Kernel, X::SubArray) = kernel_matrix(k, X[:,:], X[:,:])

kernel_matrix(k::Kernel, X::Array{Float64,1}, Xn::Array{Float64,2}) = kernel_matrix(k, X'', Xn)
kernel_matrix(k::Kernel, X::Array{Float64,2}, Xn::Array{Float64,1}) = kernel_matrix(k, X'', Xn)
kernel_matrix(k::Kernel, X::Array{Float64,1}, Xn::Array{Float64,1}) = kernel_matrix(k, X'', Xn'')
kernel_matrix(k::Kernel, X::SubArray, Xn::SubArray) = kernel_matrix(k, X[:,:], Xn[:,:])
kernel_matrix(k::Kernel, X::SubArray, Xn) = kernel_matrix(k, X[:,:], Xn)
kernel_matrix(k::Kernel, X, Xn::SubArray) = kernel_matrix(k, X, Xn[:,:])

include("rbf_kernel.jl")
include("polynomial_kernel.jl")
include("linear_kernel.jl")

kernel_from_parameters{T<:Kernel}(k::Type{T}, parameters) = k(parameters...)
kernel_from_data_model{T<:Kernel}(k::Type{T}, X) = isempty(fieldnames(k)) ? k() : k(rand(length(fieldnames(k)))...)