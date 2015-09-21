abstract Model

abstract Loss 
abstract NonParametricLoss <: Loss
immutable HINGE <: NonParametricLoss end
immutable LOGISTIC <: NonParametricLoss end
immutable SQUARED_HINGE <: NonParametricLoss end
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
immutable RK_MEANS{A <: Algorithm, M <: SemiMetric} <: Algorithm 
    support_alg::Type{A}
    k_clusters::Int
    max_iter::Int
    metric::M
end

RK_MEANS(k_clusters::Int) = RK_MEANS(PEGASOS,k_clusters,20,Euclidean())
RK_MEANS{A <: Algorithm}(support_alg::Type{A}, model::RK_MEANS) = RK_MEANS(support_alg,model.k_clusters,model.max_iter,model.metric)
RK_MEANS{M <: SemiMetric}(metric::M, model::RK_MEANS) = RK_MEANS(model.support_alg,model.k_clusters,model.max_iter,metric)

abstract Mode
immutable LINEAR <: Mode end
immutable NONLINEAR <: Mode
    k_params::Vector
    X_subset::Matrix 
end

all_algo_types = [subtypes(SGD);subtypes(RDA)]
all_loss_types = [PINBALL;subtypes(NonParametricLoss)]
create_tuple(vals, key) = (findfirst(vals,key), key)
create_tuple2(vals, key) = (findfirst(vals,key), key())

loss_opts     = Dict(map(s -> create_tuple(all_loss_types,s),        all_loss_types))
algo_opts     = Dict(map(s -> create_tuple2(all_algo_types,s),       all_algo_types))
kernel_opts   = Dict(map(s -> create_tuple(subtypes(Kernel),s),      subtypes(Kernel)))
optim_opts    = Dict(map(s -> create_tuple2(subtypes(GlobalOpt),s),  subtypes(GlobalOpt)))
criterion_opts= Dict(map(s -> create_tuple2(subtypes(CCriterion),s), subtypes(CCriterion)))
mode_opts     = Dict(map(s -> ((s == LINEAR ? 'n' : 'y'), s),        subtypes(Mode)))
loss_met_opts = Dict(1 => (LEAST_SQUARES, Euclidean()), 2 => (HINGE, CosineDist()))  

type OutputModel{M <: Mode}
    dfunc::Function
    alg_params::Vector
    X_mean::Matrix
    X_std::Matrix
    mode::M
    Ytest
    w
    b
	cv_folds::Array
	cv_n_f::Int
	cv_n::Int
	
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
    validation_criterion::Criterion
    cv_gen::@compat Nullable{CVG}
    # internals and output
    output::OutputModel{M}
end

# outer constructor to alleviate instantiation of a SLASAModel
SALSAModel{L <: Loss, A <: Algorithm, M <: Mode, K <: Kernel}(
            mode::Type{M},                      # mode used to learn model: LINEAR vs. NONLINEAR
            algorithm::A,                       # algorithm used to learn the model, e.g. PEGASOS 
            loss_function::Type{L};             # type of a loss function used to learn the model, e.g. HINGE
            kernel::Type{K} = RBFKernel,        # kernel used in NONLINEAR mode to compute Nystrom approx.
            global_opt::GlobalOpt = CSA(),      # global optimization techniques for tuning hyperparameters
            subset_size::Float64 = 5e-1,        # subset size used in NONLINEAR mode to compute Nystrom approx.
            max_cv_iter::Int = 1000,            # maximal number of iterations (budget) for any algorithm in training CV 
            max_iter::Int = 1000,               # maximal number of iterations (budget) for any algorithm for final training 
            max_cv_k::Int = 1,                  # maximal number of data points used to compute loss derivative in training CV 
            max_k::Int = 1,                     # maximal number of data points used to compute loss derivative for final training 
            online_pass::Int = 0,               # if > 0 we are in the online learning setting going through the entire dataset <online_pass> times
            normalized::Bool = true,            # normalize data (extracting mean and std) before passing it to CV and final learning 
            process_labels::Bool = true,        # process labels to comply with binary (-1 vs. 1) or multi-class classification encoding 
            tolerance::Float64 = 1e-5,          # the criterion ||w_{t+1} - w_t|| <= tolerance is evaluated for early stopping (online_pass==0) 
            sparsity_cv::Float64 = 2e-1,        # sparsity weight in the combined cross-validation/sparsity criterion used for the RDA type of algorithms 
            validation_criterion = MISCLASS(),  # validation criterion used to verify the generalization capabilities of the model in CV
            cv_gen = @compat Nullable{CrossValGenerator}()) = 
        SALSAModel(mode,algorithm,kernel,loss_function,global_opt,subset_size,max_cv_iter,max_iter,max_cv_k,max_k,
            online_pass,normalized,process_labels,tolerance,sparsity_cv,validation_criterion,cv_gen,OutputModel{mode}())

SALSAModel() = SALSAModel(LINEAR,PEGASOS(),HINGE)
SALSAModel{K <: Kernel}(kernel::Type{K}, model) = SALSAModel(model.mode,model.algorithm,model.loss_function,kernel=kernel,process_labels=model.process_labels,validation_criterion=model.validation_criterion)
SALSAModel{A <: Algorithm}(algorithm::A, model) = SALSAModel(model.mode,algorithm,model.loss_function,kernel=model.kernel,process_labels=model.process_labels,validation_criterion=model.validation_criterion)
SALSAModel{L <: Loss}(loss_function::Type{L}, model) = SALSAModel(model.mode,model.algorithm,loss_function,kernel=model.kernel,process_labels=model.process_labels,validation_criterion=model.validation_criterion)
SALSAModel{M <: Mode}(mode::Type{M}, model) = SALSAModel(mode,model.algorithm,model.loss_function,kernel=model.kernel,process_labels=model.process_labels,validation_criterion=model.validation_criterion)
SALSAModel{L <: Loss, A <: Algorithm}(loss_function::Type{L}, algorithm::A, model) = reduce((a,b)->SALSAModel(b,a),model,[loss_function,algorithm])