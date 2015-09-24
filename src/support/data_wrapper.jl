abstract DataWrapper
type DelimitedFile <: DataWrapper
	name::String
	header::Bool
	delim::Char
	stream::IOStream
end

DelimitedFile(name::String, header::Bool, delim::Char) = DelimitedFile(name, header, delim, open(name))

issparse(f::DelimitedFile) = false
count(f::DelimitedFile) = countlines(f.name) - f.header
readline(f::DelimitedFile) = begin check_stream(f); readline(f.stream) end
isempty(f::DelimitedFile) = count(f) == 0 

getindex(f::DelimitedFile, I::(@compat Tuple{Integer,Integer})) = getindex(f, I[1], I[2])

function check_stream(f::DelimitedFile)
	if eof(f.stream) 
		close(f.stream)
		f.stream = open(f.name)
		f.header ? readline(f) : Void
	end
end  

function getindex(f::DelimitedFile, i0::Integer, i1::Integer)
	map(x -> parse(Float64,x), split(readline(f),f.delim))[i1]
end

function sub(f::DelimitedFile, I::AbstractVector, ::Colon)
	vcat([map(x -> parse(Float64,x), split(readline(f),f.delim))' for i in I]...)
end

function sub(f::DelimitedFile, i::Int, ::Colon)
	map(x -> parse(Float64,x), split(readline(f),f.delim))'
end

function sub(f::DelimitedFile, ::Colon, ::Colon)
	readdlm(f.name,f.delim,header=f.header)
end

function size(f::DelimitedFile, n::Int=0)
	if n == 0
		nrows = open(countlines, f.name, "r")
		ncols = length(split(readline(f),f.delim))
		(f.header ? nrows-1 : nrows, ncols)
	elseif n == 1
		nrows = open(countlines, f.name, "r") 
		f.header ? nrows-1 : nrows
	else
		length(split(readline(f),f.delim))
	end
end