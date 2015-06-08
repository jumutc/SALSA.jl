include("enum.jl")
export CSA, DS, GlobalOpt

# enums
abstract GlobalOpt 
immutable CSA <: GlobalOpt end 
immutable DS <: GlobalOpt
	init_params::Vector 
end