module SALSA

export salsa, mapstd, make_sparse, entropysubset, AFEm, gen_cross_validate

using MLBase, Distributions, Compat


# Calculate the misclassification rate
misclass(y, yhat) = 1-mean(y.==yhat)
# Calculate the sum of squared differences between two vectors
sse(y, yhat) = norm(y-yhat)^2
# Calculates the average squared difference between the corresponding elements of two vectors
mse(y, yhat) = sse(y, yhat)/length(yhat)
# provide convenient function for parallalizing cross-validation
nfolds() = if nworkers() == 1 || nworkers() > 10 10 else nworkers() end

# needed support files
include(joinpath("kernels", "kernels.jl"))
include(joinpath("support", "constants.jl"))
include(joinpath("support", "entropysubset.jl"))
include(joinpath("support", "sparse.jl"))
include(joinpath("support", "mapstd.jl"))
include(joinpath("support", "AFEm.jl"))
include(joinpath("support", "csa.jl"))
include(joinpath("support", "ds.jl"))
# main functionality files
include("SALSAModel.jl")
include("loss_derivative.jl")
include("model_ext.jl")
include("predict.jl")
# main algorithmic files
include(joinpath("algorithms", "l1rda_alg.jl"))
include(joinpath("algorithms", "adaptive_l1rda_alg.jl"))
include(joinpath("algorithms", "reweighted_l1rda_alg.jl"))
include(joinpath("algorithms", "reweighted_l2rda_alg.jl"))
include(joinpath("algorithms", "stochastic_k_means.jl"))
include(joinpath("algorithms", "pegasos_alg.jl"))
include(joinpath("algorithms", "dropout_alg.jl"))
# tuning + validation
include("run_algorithm.jl")
include("validation_criteria.jl")
include("tune_algorithm.jl")
include("tune_algorithm_AFEm.jl")

# extensive set of multiplicated aliases for different algorithms and models /// dense matrices
salsa{L <: Loss, A <: Algorithm, M <: Mode}(mode::Type{M}, alg::Type{A}, loss::Type{L}, X::Array{Float64,2}, Y::Array{Float64,1}, Xtest::Array{Float64,2}) = salsa(X,Y,SALSAModel(mode,alg(),loss),Xtest)
salsa{L <: Loss, A <: Algorithm, M <: Mode}(mode::Type{M}, alg::Type{A}, loss::Type{L}, X::Array{Float64,2}, Y::Array{Float64,2}, Xtest::Array{Float64,2}) = salsa(X,Y,SALSAModel(mode,alg(),loss),Xtest)
salsa{A <: Algorithm}(alg::Type{A}, X::Array{Float64,2}, Y::Array{Float64,1}, Xtest::Array{Float64,2}) = salsa(X,Y,SALSAModel(LINEAR,alg(),HINGE),Xtest)
salsa{A <: Algorithm}(alg::Type{A}, X::Array{Float64,2}, Y::Array{Float64,2}, Xtest::Array{Float64,2}) = salsa(X,Y,SALSAModel(LINEAR,alg(),HINGE),Xtest)
salsa(X::Array{Float64,2}, Y::Array{Float64,1}, Xtest::Array{Float64,2}) = salsa(LINEAR,PEGASOS,HINGE,X,Y,Xtest)
salsa(X::Array{Float64,2}, Y::Array{Float64,2}, Xtest::Array{Float64,2}) = salsa(LINEAR,PEGASOS,HINGE,X,Y,Xtest)
salsa(X::Array{Float64,2}, Y::Array{Float64,1}) = salsa(LINEAR,PEGASOS,HINGE,X,Y,Array{Float64}(0,0))
salsa(X::Array{Float64,2}, Y::Array{Float64,2}) = salsa(LINEAR,PEGASOS,HINGE,X,Y,Array{Float64}(0,0))
# extensive set of multiplicated aliases for different algorithms and models /// sparse matrices
salsa{L <: Loss, A <: Algorithm, M <: Mode}(mode::Type{M}, alg::Type{A}, loss::Type{L}, X::SparseMatrixCSC, Y::Array{Float64,1}, Xtest::SparseMatrixCSC) = salsa(X,Y,SALSAModel(mode,alg(),loss),Xtest)
salsa{L <: Loss, A <: Algorithm, M <: Mode}(mode::Type{M}, alg::Type{A}, loss::Type{L}, X::SparseMatrixCSC, Y::Array{Float64,2}, Xtest::SparseMatrixCSC) = salsa(X,Y,SALSAModel(mode,alg(),loss),Xtest)
salsa{A <: Algorithm}(alg::Type{A}, X::SparseMatrixCSC, Y::Array{Float64,1}, Xtest::SparseMatrixCSC) = salsa(X,Y,SALSAModel(LINEAR,alg(),HINGE),Xtest)
salsa{A <: Algorithm}(alg::Type{A}, X::SparseMatrixCSC, Y::Array{Float64,2}, Xtest::SparseMatrixCSC) = salsa(X,Y,SALSAModel(LINEAR,alg(),HINGE),Xtest)
salsa(X::SparseMatrixCSC, Y::Array{Float64,1}, Xtest::SparseMatrixCSC) = salsa(LINEAR,PEGASOS,HINGE,X,Y,Xtest)
salsa(X::SparseMatrixCSC, Y::Array{Float64,2}, Xtest::SparseMatrixCSC) = salsa(LINEAR,PEGASOS,HINGE,X,Y,Xtest)
salsa(X::SparseMatrixCSC, Y::Array{Float64,1}) = salsa(LINEAR,PEGASOS,HINGE,X,Y,sparse([]))
salsa(X::SparseMatrixCSC, Y::Array{Float64,2}) = salsa(LINEAR,PEGASOS,HINGE,X,Y,sparse([]))


function gen_cross_validate(evalfun::Function, X, Y, model::SALSAModel)
	indices = get(model.cv_gen, Kfold(length(Y),nfolds()))
    @parallel (+) for train_idx in collect(indices)
		val_idx = setdiff(1:length(Y), train_idx)
        evalfun(X[train_idx,:], Y[train_idx], X[val_idx,:], Y[val_idx])/nfolds()
    end
end

function gen_cross_validate(evalfun::Function, n::Int64, model::SALSAModel)
	indices = get(model.cv_gen, Kfold(n,nfolds())) 
    @parallel (+) for train_idx in collect(indices)
		val_idx = setdiff(1:n, train_idx)
        evalfun(train_idx, val_idx)/nfolds()
    end
end


# External function for a complete stochastic learning routine with cross-validation
	#
	# Copyright (c) 2015, KU Leuven-ESAT-STADIUS, License & help @
	# http://www.esat.kuleuven.be/stadius/ADB/software.php
	#
	#    model = salsa(X,Y,model,Xtest) runs Pegasos, RDA... stochastic
	#    algorithms suited for large-scale binary data classification
	#
	# Based on:
	# 
    # [1] S. Shalev-Shwartz, Y. Singer, N. Srebro, Pegasos: Primal Estimated sub-GrAdient SOlver for SVM, 
    # in: Proceedings of the 24th international conference on Machine learning, ICML ’07, New York, NY, USA, 2007, pp. 807–814.

    # [2] L. Xiao, Dual averaging methods for regularized stochastic learning and online optimization, 
    # J. Mach. Learn. Res. 11 (2010) 2543–2596.

    # [3] J. Duchi, E. Hazan, Y. Singer, Adaptive subgradient methods for online learning and stochastic optimization, 
    # J. Mach. Learn. Res. 12 (2011) 2121–2159.

    # [4] V. Jumutc, J. A. K. Suykens, Reweighted l1 dual averaging approach for sparse stochastic learning, 
    # in: 22th European Symposium on Artificial Neural Networks, ESANN 2014, Bruges, Belgium, April 23-25, 2014.

    # [5] V. Jumutc, J. A. K. Suykens, Reweighted l2 -regularized dual averaging approach for highly sparse stochastic learning, 
    # in: Advances in Neural Networks - 11th International Symposium on Neural Networks, ISNN 2014, 
    # Hong Kong and Macao, China, November 28 – December 1, 2014, pp. 232–242.
	
function salsa(X, Y, model::SALSAModel, Xtest)
	if ~isdefined(model,:output)
		model.output = OutputModel{model.mode}()
	end

    if model.normalized && isempty(Xtest) 
	    (X, model.output.X_mean, model.output.X_std) = mapstd(X)
	elseif model.normalized
	    (X, model.output.X_mean, model.output.X_std) = mapstd(X)
	    Xtest = mapstd(Xtest,model.output.X_mean,model.output.X_std)
	end

	if model.mode == LINEAR
	    model = tune_algorithm(X,Y,model)
	    (model.output.w, model.output.b) = run_algorithm(X,Y,model)
	else
	    model = tune_algorithm_AFEm(X,Y,model) 
	    # find actual Nystrom-approximated feature map and run Pegasos
	    k = kernel_from_parameters(model.kernel,model.output.mode.k_params)
	    features = AFEm(model.output.mode.X_subset,k,X)
	    (model.output.w, model.output.b) = run_algorithm(features,Y,model)	    
	end

	if !isempty(Xtest)
	    model.output.Ytest = predict(model,Xtest)
	end

	model
end

end