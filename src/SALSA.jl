module SALSA

export salsa, 
	salsa_qa,
	# preprocessing routines
	mapstd, 
	make_sparse,
	# Nystrom approximation 
	entropy_subset, 
	AFEm,
	LINEAR, 
	NONLINEAR,
	# cross-validation 
	gen_cross_validate, 
	mse, 
	misclass, 
	auc,
	# SALSA model 
	SALSAModel,
	# loss functions 
    HINGE,
    PINBALL, 
    LOGISTIC, 
    LEAST_SQUARES, 
    MODIFIED_HUBER,
    # algorithms
    SIMPLE_SGD, 
    PEGASOS, 
    L1RDA, 
    ADA_L1RDA, 
    R_L1RDA, 
    R_L2RDA, 
    DROP_OUT, 
    RK_MEANS,
    # support
    predict, 
    predict_latent, 
    map_predict, 
    map_predict_latent, 
    predict_by_distance,
    loss_derivative,
    membership,
    DelimitedFile,
    # global optimization
    CSA, DS, GlobalOpt,
    # cross-validation criteria
	MISCLASS, 
	AUC, MSE, 
	SILHOUETTE, 
	# kernels for Nystrom approximation 
	RBFKernel, 
	LinearKernel, 
	PolynomialKernel,
	# core algorithmic schemas
	adaptive_l1rda_alg,
	reweighted_l1rda_alg,
	reweighted_l2rda_alg,
	pegasos_alg,
	dropout_alg,
	l1rda_alg,
	sgd_alg,
	stochastic_rk_means


using MLBase, Distributions, Compat, Distances, Clustering
import Base: size, getindex, issparse, sub
import StatsBase: counts

# needed support files
include(joinpath("kernels", "kernels.jl"))
include(joinpath("support", "constants.jl"))
include(joinpath("support", "entropy_subset.jl"))
include(joinpath("support", "algorithm_support.jl"))
include(joinpath("support", "validation_support.jl"))
include(joinpath("support", "data_wrapper.jl"))
include(joinpath("support", "membership.jl"))
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
# needed support files for cross-validation
include(joinpath("support", "cross_validation.jl"))
# Q/A tables 
include(joinpath("qa_tables", "salsa_qa.jl"))
# main algorithmic files
include(joinpath("algorithms", "l1rda_alg.jl"))
include(joinpath("algorithms", "adaptive_l1rda_alg.jl"))
include(joinpath("algorithms", "reweighted_l1rda_alg.jl"))
include(joinpath("algorithms", "reweighted_l2rda_alg.jl"))
include(joinpath("algorithms", "stochastic_rk_means.jl"))
include(joinpath("algorithms", "pegasos_alg.jl"))
include(joinpath("algorithms", "dropout_alg.jl"))
include(joinpath("algorithms", "sgd_alg.jl"))
# tuning + validation
include("run_algorithm.jl")
include("validation_criteria.jl")
include("tune_algorithm.jl")
include("tune_algorithm_AFEm.jl")
# main runnable
include("salsa.jl")

# extensive set of multiplicated aliases for different algorithms and models /// dense matrices
salsa{L <: Loss, A <: Algorithm, M <: Mode, N1 <: Number, N2 <: Number}(mode::Type{M}, alg::Type{A}, loss::Type{L}, X::Array{N1,2}, Y::Array{N2,1}, Xtest::Array{N1,2}) = salsa(X,Y,SALSAModel(mode,alg(),loss),Xtest)
salsa{L <: Loss, A <: Algorithm, M <: Mode, N1 <: Number, N2 <: Number}(mode::Type{M}, alg::Type{A}, loss::Type{L}, X::Array{N1,2}, Y::Array{N2,2}, Xtest::Array{N1,2}) = salsa(X,Y,SALSAModel(mode,alg(),loss),Xtest)
salsa{A <: Algorithm, N1 <: Number, N2 <: Number}(alg::Type{A}, X::Array{N1,2}, Y::Array{N2,1}, Xtest::Array{N1,2}) = salsa(X,Y,SALSAModel(LINEAR,alg(),HINGE),Xtest)
salsa{A <: Algorithm, N1 <: Number, N2 <: Number}(alg::Type{A}, X::Array{N1,2}, Y::Array{N2,2}, Xtest::Array{N1,2}) = salsa(X,Y,SALSAModel(LINEAR,alg(),HINGE),Xtest)
salsa{N1 <: Number, N2 <: Number}(X::Array{N1,2}, Y::Array{N2,1}, Xtest::Array{N1,2}) = salsa(LINEAR,PEGASOS,HINGE,X,Y,Xtest)
salsa{N1 <: Number, N2 <: Number}(X::Array{N1,2}, Y::Array{N2,2}, Xtest::Array{N1,2}) = salsa(LINEAR,PEGASOS,HINGE,X,Y,Xtest)
salsa{N1 <: Number, N2 <: Number}(X::Array{N1,2}, Y::Array{N2,1}) = salsa(LINEAR,PEGASOS,HINGE,X,Y,Array{Int64}(0,0))
salsa{N1 <: Number, N2 <: Number}(X::Array{N1,2}, Y::Array{N2,2}) = salsa(LINEAR,PEGASOS,HINGE,X,Y,Array{Int64}(0,0))
# extensive set of multiplicated aliases for different algorithms and models /// sparse matrices
salsa{L <: Loss, A <: Algorithm, M <: Mode, N <: Number}(mode::Type{M}, alg::Type{A}, loss::Type{L}, X::SparseMatrixCSC, Y::Array{N,1}, Xtest::SparseMatrixCSC) = salsa(X,Y,SALSAModel(mode,alg(),loss),Xtest)
salsa{L <: Loss, A <: Algorithm, M <: Mode, N <: Number}(mode::Type{M}, alg::Type{A}, loss::Type{L}, X::SparseMatrixCSC, Y::Array{N,2}, Xtest::SparseMatrixCSC) = salsa(X,Y,SALSAModel(mode,alg(),loss),Xtest)
salsa{A <: Algorithm, N <: Number}(alg::Type{A}, X::SparseMatrixCSC, Y::Array{N,1}, Xtest::SparseMatrixCSC) = salsa(X,Y,SALSAModel(LINEAR,alg(),HINGE),Xtest)
salsa{A <: Algorithm, N <: Number}(alg::Type{A}, X::SparseMatrixCSC, Y::Array{N,2}, Xtest::SparseMatrixCSC) = salsa(X,Y,SALSAModel(LINEAR,alg(),HINGE),Xtest)
salsa{N <: Number}(X::SparseMatrixCSC, Y::Array{N,1}, Xtest::SparseMatrixCSC) = salsa(LINEAR,PEGASOS,HINGE,X,Y,Xtest)
salsa{N <: Number}(X::SparseMatrixCSC, Y::Array{N,2}, Xtest::SparseMatrixCSC) = salsa(LINEAR,PEGASOS,HINGE,X,Y,Xtest)
salsa{N <: Number}(X::SparseMatrixCSC, Y::Array{N,1}) = salsa(LINEAR,PEGASOS,HINGE,X,Y,sparse([]))
salsa{N <: Number}(X::SparseMatrixCSC, Y::Array{N,2}) = salsa(LINEAR,PEGASOS,HINGE,X,Y,sparse([]))

end