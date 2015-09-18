# fix for julia release where this function is absent, TODO: remove when we move to julia 0.4+
sub(a::SparseMatrixCSC, I::AbstractVector, ::Colon) = sparse(a[I,:])
sub(a::SubArray, I::AbstractVector, ::Colon) = convert(Array, a[I,:])
sub(a::AbstractMatrix, I::AbstractVector, ::Colon) = a[I,:]
sub(a::AbstractVector, I::AbstractVector, ::Colon) = a[I]
sub(a::AbstractMatrix, i::Int, ::Colon) = a[i,:]
sub(a::AbstractMatrix, ::Colon, ::Colon) = a
view(a::SparseMatrixCSC, ::Colon, I::Int) = a[:,I]
dot(a::SparseMatrixCSC, b::SparseMatrixCSC) = sum(a.*b)