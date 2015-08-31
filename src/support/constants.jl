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

DS() = DS(Array(Float64,0))
AUC() = AUC(100)

show(io::IO, t::MSE) = @printf io "%s (%s)" typeof(t) "Mean Squared Error"
show(io::IO, t::MISCLASS) = @printf io "%s (%s)" typeof(t) "Misclassification Rate"
show(io::IO, t::SILHOUETTE) = @printf io "%s (%s)" typeof(t) "Silhouette Index"
show(io::IO, t::AUC) = @printf io "%s (%s with %d thresholds)" typeof(t) "Area Under ROC Curve" t.n_thresholds
show(io::IO, t::CSA) = @printf io "%s (%s)" typeof(t) "Coupled Simulated Annealing"
show(io::IO, t::DS) = @printf io "%s (%s)" typeof(t) "Directional Search"