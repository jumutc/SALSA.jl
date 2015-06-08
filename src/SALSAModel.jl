export SALSAModel, 
       HINGE, PINBALL, LOGISTIC, LEAST_SQUARES,
       PEGASOS, L1RDA, ADA_L1RDA, R_L1RDA, R_L2RDA, DROP_OUT, RDA, SGD, K_MEANS,
       LINEAR, NONLINEAR

abstract Model

abstract Loss 
abstract NonParametricLoss <: Loss
immutable HINGE <: NonParametricLoss end
immutable LOGISTIC <: NonParametricLoss end
immutable LEAST_SQUARES <: NonParametricLoss end
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
# special algorithm type for clustering
immutable K_MEANS{A <: Algorithm} <: Algorithm 
    support_alg::Type{A}
    k_clusters::Int
end

abstract Mode
immutable LINEAR <: Mode end
immutable NONLINEAR <: Mode
    k_params::Vector
    X_subset::Matrix 
end

type OutputModel{M <: Mode}
    dfunc::Function
    alg_params::Vector
    X_mean::Matrix
    X_std::Matrix
    mode::M
    Ytest
    w
    b

    OutputModel() = new()
end

type SALSAModel{L <: Loss, A <: Algorithm, 
                M <: Mode, K <: Kernel, 
                CVG <: CrossValGenerator} <: Model
    mode::Type{M}
    algorithm::A
    kernel::Type{K}
    loss_function::Type{L}
    global_opt::GlobalOpt
    subset_size::Float64
    max_cv_iter::Int
    max_iter::Int 
    max_cv_k::Int 
    max_k::Int
    online_pass::Bool
    normalized::Bool
    tolerance::Float64
    sparsity_cv::Float64
    cv_gen::@compat Nullable{CVG}
    
    # internals and output
    output::OutputModel{M}
end

# outer constructor to alleviate instantiation of a SLASAModel
SALSAModel{L <: Loss, A <: Algorithm, M <: Mode, K <: Kernel}(
            mode::Type{M},
            alg::Type{A},
            loss_function::Type{L};
            kernel::Type{K} = RBFKernel,
            global_opt::GlobalOpt = CSA(),
            subset_size::Float64 = 5e-1,
            max_cv_iter::Int = 1000,
            max_iter::Int = 1000,
            max_cv_k::Int = 1,
            max_k::Int = 1,
            online_pass::Bool = false,
            normalized::Bool = true,
            tolerance::Float64 = 1e-5,
            sparsity_cv::Float64 = 2e-2,
            cv_gen = @compat Nullable{CrossValGenerator}()) = 
        SALSAModel(mode,alg(),kernel,loss_function,global_opt,subset_size,max_cv_iter,
                   max_iter,max_cv_k,max_k,online_pass,normalized,tolerance,sparsity_cv,cv_gen,OutputModel{mode}())