abstract Model

abstract Loss 
abstract NonParametricLoss <: Loss
immutable HINGE <: NonParametricLoss end
immutable LOGISTIC <: NonParametricLoss end
immutable MODIFIED_HUBER <: NonParametricLoss end
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
immutable SIMPLE_SGD <: SGD end
# special algorithm type for Regularized K-Means
immutable RK_MEANS{A <: Algorithm, M <: Metric} <: Algorithm 
    support_alg::Type{A}
    k_clusters::Int
    max_iter::Int
    metric::M
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
    online_pass::Int
    normalized::Bool
    process_labels::Bool
    tolerance::Float64
    sparsity_cv::Float64
    validation_criteria::Criteria
    cv_gen::@compat Nullable{CVG}
    
    # internals and output
    output::OutputModel{M}
end

# outer constructor to alleviate instantiation of a SLASAModel
SALSAModel{L <: Loss, A <: Algorithm, M <: Mode, K <: Kernel}(
            mode::Type{M},                      # mode used to learn model: LINEAR vs. NONLINEAR
            algorithm::A,                       # algorithm used to learn the model, e.g. PEGASOS 
            loss_function::Type{L};             # type of a loss function used to learn model, e.g. HINGE
            kernel::Type{K} = RBFKernel,        # kernel used in NONLINEAR mode to compute Nystrom approx.
            global_opt::GlobalOpt = CSA(),      # global optimization techniques for tuning hyperparameters
            subset_size::Float64 = 5e-1,        # subset size used in NONLINEAR mode to compute Nystrom approx.
            max_cv_iter::Int = 1000,            # maximal number of iterations (budget) for any algorithm in training CV 
            max_iter::Int = 1000,               # maximal number of iterations (budget) for any algorithm for final training 
            max_cv_k::Int = 1,                  # maximal number of data points used to compute loss derivative in training CV 
            max_k::Int = 1,                     # maximal number of data points used to compute loss derivative for final training 
            online_pass::Int = 0,               # if > 0 we are in the online learning setting going through entire dataset <online_pass> times
            normalized::Bool = true,            # normalize data (extracting mean and std) before passing it to CV and final learning 
            process_labels::Bool = true,        # process labels to comply with binary (-1 vs. 1) or multi-class classification encoding 
            tolerance::Float64 = 1e-5,          # criteria ||w_{t+1} - w_t|| <= tolerance is evaluated for early stopping (online_pass==0) 
            sparsity_cv::Float64 = 2e-2,        # sparisty affinity compelment to any validation_criteria for CV used in RDA type of algorithms 
            validation_criteria = MISCLASS(),   # validation criteria used to verify the generalization capabilities of the model in CV
            cv_gen = @compat Nullable{CrossValGenerator}()) = 
        SALSAModel(mode,algorithm,kernel,loss_function,global_opt,subset_size,max_cv_iter,max_iter,max_cv_k,max_k,
            online_pass,normalized,process_labels,tolerance,sparsity_cv,validation_criteria,cv_gen,OutputModel{mode}())