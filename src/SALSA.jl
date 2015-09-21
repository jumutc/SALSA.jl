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
    SQUARED_HINGE,
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
    loss_derivative,
    membership,
    DelimitedFile,
    # global optimization
    CSA, DS, GlobalOpt,
    csa, ds,
    # cross-validation criterion
	MISCLASS, 
	AUC, MSE, 
	SILHOUETTE, 
	# kernels for Nystrom approximation 
	RBFKernel, 
	LinearKernel, 
	PolynomialKernel,
	# core algorithmic schemas
	stochastic_rk_means,
	adaptive_l1rda_alg,
	reweighted_l1rda_alg,
	reweighted_l2rda_alg,
	pegasos_alg,
	dropout_alg,
	l1rda_alg,
	sgd_alg


using MLBase, Distributions, Compat, Distances, Clustering
import Base: size, getindex, issparse, sub, dot, show, isempty, At_mul_B!
import StatsBase: counts, predict
import ArrayViews: view

# needed support files
include(joinpath("support", "data_wrapper.jl"))
include(joinpath("support", "definitions.jl"))
include(joinpath("kernels", "kernels.jl"))
include(joinpath("support", "constants.jl"))
include(joinpath("support", "entropy_subset.jl"))
include(joinpath("support", "algorithm_support.jl"))
include(joinpath("support", "validation_support.jl"))
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
include(joinpath("qa_tables", "QAModel.jl"))
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
# main runnable source
include("salsa_main.jl")
# fine printing out
include("print.jl")

end