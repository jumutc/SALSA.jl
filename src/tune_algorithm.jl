function tune_algorithm(X, Y, model::SALSAModel)
    cost_fun = x0 -> cross_validate_algorithm(x0,X,Y,model)
    par = run_global_opt(model,cost_fun, model.global_opt)
    
    # generate model from the parameters
    model.output.mode = LINEAR()
    model_from_parameters(model,par)
end

function cross_validate_algorithm(x0, X, Y, model)
    # generate model from parameters
    model = model_from_parameters(model,x0)
    # perform Kfold cross-validation by a generic and parallelizable function
    gen_cross_validate(length(Y), model) do train_idx, val_idx    
        # run Pegasos algorithm for the excluded subset of validation indices        
        (model.output.w, model.output.b) = run_algorithm(X,Y,model,train_idx)
        validation_criteria(model,X,Y,val_idx)
    end
end

function run_global_opt(model::SALSAModel, cost_fun::Function, global_opt::CSA)
    # Coupled Simulated Annealing calculations
    eval_fun = pars -> [cost_fun(pars[:,i]) for i=1:size(pars,2)]
    (fval, par) = csa(eval_fun, randn(5,5))
    @printf "CSA results: optimal %s = %.3f\n" validation_criteria(model) fval
    return par
end 

function run_global_opt(model::SALSAModel, cost_fun::Function, global_opt::DS)
    # Randomized Directional Search calculations
    (fval, par) = ds(cost_fun, global_opt.init_params)
    @printf "DS results: optimal %s = %.3f\n" validation_criteria(model) fval
    return par
end 
