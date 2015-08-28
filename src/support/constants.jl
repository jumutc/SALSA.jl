abstract Criteria 
abstract CCriteria <: Criteria
immutable MSE <: CCriteria end
immutable MISCLASS <: CCriteria end 
immutable SILHOUETTE <: Criteria end 
immutable AUC <: CCriteria
	n_thresholds::Integer
end

abstract GlobalOpt 
immutable CSA <: GlobalOpt end 
immutable DS <: GlobalOpt
	init_params::Vector 
end

DS() = DS([0])
AUC() = AUC(100)