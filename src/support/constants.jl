abstract Criterion 
abstract CCriterion <: Criterion
immutable MSE <: CCriterion end
immutable MISCLASS <: CCriterion end 
immutable SILHOUETTE <: Criterion end 
immutable AUC <: CCriterion
	n_thresholds::Integer
end

abstract GlobalOpt 
immutable CSA <: GlobalOpt end 
immutable DS <: GlobalOpt
	init_params::Vector 
end

DS() = DS([0])
AUC() = AUC(100)