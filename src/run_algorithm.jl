# extensive set of multiplicated aliases for different algorithms 
run_algorithm{L <: Loss, M <: Mode}(X, Y, model::SALSAModel{L, PEGASOS, M}, train_idx=[]) 		= pegasos_alg(model.output.dfunc, X, Y, model.output.alg_params..., model.max_cv_k, model.max_cv_iter, model.tolerance, model.online_pass, train_idx) 
run_algorithm{L <: Loss, M <: Mode}(X, Y, model::SALSAModel{L, SIMPLE_SGD, M}, train_idx=[])	= sgd_alg(model.output.dfunc, X, Y, model.output.alg_params..., model.max_cv_k, model.max_cv_iter, model.tolerance, model.online_pass, train_idx) 
run_algorithm{L <: Loss, M <: Mode}(X, Y, model::SALSAModel{L, L1RDA, M}, train_idx=[]) 		= l1rda_alg(model.output.dfunc, X, Y, model.output.alg_params..., model.max_cv_k, model.max_cv_iter, model.tolerance, model.online_pass, train_idx) 
run_algorithm{L <: Loss, M <: Mode}(X, Y, model::SALSAModel{L, R_L1RDA, M}, train_idx=[]) 		= reweighted_l1rda_alg(model.output.dfunc, X, Y, model.output.alg_params..., model.max_cv_k, model.max_cv_iter, model.tolerance, model.online_pass, train_idx) 
run_algorithm{L <: Loss, M <: Mode}(X, Y, model::SALSAModel{L, R_L2RDA, M}, train_idx=[]) 		= reweighted_l2rda_alg(model.output.dfunc, X, Y, model.output.alg_params..., model.max_cv_k, model.max_cv_iter, model.tolerance, model.online_pass, train_idx) 
run_algorithm{L <: Loss, M <: Mode}(X, Y, model::SALSAModel{L, DROP_OUT, M}, train_idx=[]) 		= dropout_alg(model.output.dfunc, X, Y, model.output.alg_params..., model.max_cv_k, model.max_cv_iter, model.tolerance, model.online_pass, train_idx)
run_algorithm{L <: Loss, M <: Mode}(X, Y, model::SALSAModel{L, ADA_L1RDA, M}, train_idx=[]) 	= adaptive_l1rda_alg(model.output.dfunc, X, Y, model.output.alg_params..., model.max_cv_k, model.max_cv_iter, model.tolerance, model.online_pass, train_idx)
run_algorithm{L <: Loss, A <: RK_MEANS, M <: Mode}(X, Y, model::SALSAModel{L,A,M}, train_idx=[])= (stochastic_rk_means(X, model.algorithm, model.output.alg_params, model.max_cv_k, model.max_cv_iter, model.tolerance, model.online_pass, train_idx), 0.0)
partition_pars(pars,k) = 5*k > length(pars) ? pars[5*(k-1)+1:end] : pars[5*(k-1)+1:5*k]

function run_with_params{L <: Loss, A <: RK_MEANS, M <: Mode}(X, Y, model::SALSAModel{L,A,M}, pars, train_idx=[]) 
	model = model_from_parameters(model,pars)
	run_algorithm(X,Y,model,train_idx)
end
function run_with_params(X, Y, model::SALSAModel, pars, train_idx=[])
	w_ = zeros(size(X,2),size(Y,2)); b_ = zeros(size(Y,2))'

    for k in 1:size(Y,2)
        # generate model from the partitioned parameters
        model = model_from_parameters(model,partition_pars(pars,k))
        # run algorithm for the excluded subset of validation indices        
        w_[:,k], b_[:,k] = run_algorithm(X,Y[:,k],model,train_idx)
    end

    (w_, b_)
end