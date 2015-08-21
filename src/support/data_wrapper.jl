using DataArrays, DataFrames

abstract DataWrapper
immutable DelimitedFile <: DataWrapper
	name::ASCIIString
	header::Bool
	delim::Char
end

issparse(f::DelimitedFile) = false

getindex(f::DelimitedFile, I::(@compat Tuple{Integer,Integer})) = getindex(f, I[1], I[2])

function getindex(f::DelimitedFile, i0::Integer, i1::Integer)
	readtable(f.name, separator=f.delim, skipstart=(i0-1), nrows=1, header=f.header)[1,i1]
end

function sub(f::DelimitedFile, I::AbstractVector, ::Colon)
	vcat([convert(Array, readtable(f.name, separator=f.delim, skipstart=(i-1), nrows=1, header=f.header)) for i in I]...)
end

# fix for julia release where this function is absent, TODO: remove when we move to julia 0.4-
sub(a::SubArray, I::AbstractVector, ::Colon) = convert(Array, a[I,:])
sub(a::Matrix, 	 I::AbstractVector, ::Colon) = a[I,:]

function size(f::DelimitedFile, n::Int=0)
	if n == 0
		nrows = open(countlines, f.name, "r")
		ncols = size(readtable(f.name, separator=f.delim, nrows=1, header=f.header), 2)
		(f.header ? nrows-1 : nrows, ncols)
	elseif n == 1
		nrows = open(countlines, f.name, "r")
		f.header ? nrows-1 : nrows
	else
		size(readtable(f.name, separator=f.delim, nrows=1, header=f.header), 2)
	end
end