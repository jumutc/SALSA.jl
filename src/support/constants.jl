include("enum.jl")
export CSA, DS, GlobalOpt,
	   MISCLASS, AUC, Criteria

# enums
abstract Criteria 
immutable MISCLASS <: Criteria end 
immutable AUC <: Criteria
	n_thresholds::Integer
end

abstract GlobalOpt 
immutable CSA <: GlobalOpt end 
immutable DS <: GlobalOpt
	init_params::Vector 
end