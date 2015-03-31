export SALSAModel, 
       HINGE, PINBALL, LOGISTIC,
       PEGASOS, L1RDA, ADA_L1RDA, R_L1RDA, R_L2RDA, DROP_OUT, RDA, SGD,
       LINEAR, NONLINEAR

abstract Model

abstract Loss 
abstract NonParametricLoss <: Loss
immutable HINGE <: NonParametricLoss end
immutable LOGISTIC <: NonParametricLoss end
immutable PINBALL <: Loss end

abstract Algorithm 
abstract RDA <: Algorithm
abstract SGD <: Algorithm
immutable PEGASOS <: SGD end
immutable L1RDA <: RDA end
immutable R_L1RDA <: RDA end
immutable R_L2RDA <: RDA end
immutable ADA_L1RDA <: RDA end
immutable DROP_OUT <: SGD end

abstract Mode
immutable LINEAR <: Mode end
immutable NONLINEAR{K <: Kernel} <: Mode end

type SALSAModel{L <: Loss, A <: Algorithm, M <: Mode, K <: Kernel} <: Model
    mode::Type{M}
    algorithm::Type{A}
    loss_function::Type{L}
    global_opt::GlobalOpt
    subset_size::Float64
    num_cv_folds::Int
    max_cv_iter::Int
    max_iter::Int 
    max_cv_k::Int 
    max_k::Int
    online_pass::Bool
    normalized::Bool
    tolerance::Float64
    sparsity_cv::Float64
    kernel::Type{K}
    # internals and output
    subset::YVar
    dfunc::Function
    alg_params::YVar
    k_params::YVar
    X_mean
    X_std
    Ytest
    w
    b
     
    SALSAModel() = new(M,A,L,CSA,5e-1,nfolds(),1000,1000,1,1,false,true,1e-5,2e-1,K)
end