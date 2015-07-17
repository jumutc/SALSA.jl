include("enum.jl")
export CSA, DS, GlobalOpt,
	   MISCLASS, AUC, MSE, SILHOUETTE, Criteria

# enums
abstract Criteria 
immutable MSE <: Criteria end
immutable SILHOUETTE <: Criteria end 
immutable MISCLASS <: Criteria end 
immutable AUC <: Criteria
	n_thresholds::Integer
end

abstract GlobalOpt 
immutable CSA <: GlobalOpt end 
immutable DS <: GlobalOpt
	init_params::Vector 
end