function tune_algorithm(X, Y, model::SALSAModel)
    cost_fun = x0 -> cross_validate_algorithm(x0,X,Y,model)
    eval_fun = pars -> [cost_fun(pars[:,i]) for i=1:size(pars,2)]

    if model.global_opt == CSA 
        # Coupled Simulated Annealing calculations
        (fval, par) = csa(eval_fun, randn(5,5))
        @printf "CSA results: fval=%.5f\n" fval 
    elseif model.global_opt == DS 
        # Randomized Directional Search calculations
        init_params = ds_parameters_from_model(model)
        (fval, par) = ds(cost_fun, init_params)
        @printf "DS results: fval=%.5f\n" fval 
    else
        error("Please specify model.global_opt")
    end
    
    # generate model from the parameters
    model.output.mode = LINEAR()
    model_from_parameters(model,par)
end

function cross_validate_algorithm(x0, X, Y, model)
    # generate model from parameters
    model = model_from_parameters(model,x0)
    # perform Kfold cross-validation by a generic and parallelizable function
    gen_cross_validate(length(Y), model.cv_gen) do train_idx, val_idx    
        # run Pegasos algorithm for the excluded subset of validation indices        
        (model.output.w, model.output.b) = run_algorithm(X,Y,model,train_idx)
        validation_criteria(model,X,Y,val_idx)
    end
end