function tune_algorithm(X, Y, model::SALSAModel)
    cost_fun = x0 -> cross_validate_algorithm(x0,X,Y,model)
    par = run_global_opt(model,cost_fun,model.global_opt,(size(Y,2)*5,5))
    
    # return model and the parameters
    model.output.mode = LINEAR()
    model, par
end

function cross_validate_algorithm(x0, X, Y, model)
    # perform cross-validation by a generic and parallelizable function
    gen_cross_validate(size(Y,1), model) do train_idx, val_idx
        w_ = zeros(size(X,2),size(Y,2)); b_ = zeros(size(Y,2))'

        for k in 1:size(Y,2)
            # generate model from the partitioned parameters
            model = model_from_parameters(model,partition_pars(x0,k))
            # run algorithm for the excluded subset of validation indices        
            w_[:,k], b_[:,k] = run_algorithm(X,Y[:,k],model,train_idx)
        end
        
        model.output.w = w_; model.output.b = b_ 
        validation_criteria(model,X,Y,val_idx)
    end
end

function run_global_opt(model::SALSAModel, cost_fun::Function, global_opt::CSA, par_dims)
    # Coupled Simulated Annealing calculations
    eval_fun = pars -> [cost_fun(pars[:,i]) for i=1:size(pars,2)]
    (fval, par) = csa(eval_fun, randn(par_dims))
    @printf "CSA results: optimal %s = %.3f\n" validation_criteria(model.validation_criteria, model) fval
    return par
end 

function run_global_opt(model::SALSAModel, cost_fun::Function, global_opt::DS, par_dims)
    # Randomized Directional Search calculations
    params = global_opt.init_params
    (fval, par) = ds(cost_fun, isempty(params) ? randn(par_dims[1]) : params)
    @printf "DS results: optimal %s = %.3f\n" validation_criteria(model.validation_criteria, model) fval
    return par
end 

partition_pars(pars,k) = 5*k > length(pars) ? pars[5*(k-1)+1:end] : pars[5*(k-1)+1:5*k]