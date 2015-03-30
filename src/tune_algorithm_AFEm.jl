function tune_algorithm_AFEm(X, Y, model::SALSAModel)
    # Renyi entropy calculations
    k = kernel_from_data_model(model.kernel,X)    
    rp = ceil(sqrt(size(X,1)*model.subset_size))
    model.subset = entropysubset(X,k,rp)
    num_k = length(model.kernel.names)
    
    cost_fun = x0 -> cross_validate_algorithm_AEFm(x0,X,Y,model,num_k)
    eval_fun = pars -> [cost_fun(pars[:,i]) for i=1:size(pars,2)]

    if model.global_opt == CSA 
        # Coupled Simulated Annealing calculations
        (fval, par) = csa(eval_fun, randn(5+num_k,5))
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
    model.k_params = exp(par[end-num_k+1:end])
    model_from_parameters(model,par)
end

function cross_validate_algorithm_AEFm(x0, X, Y, model, num_k)
    k = kernel_from_parameters(model.kernel,exp(x0[end-num_k+1:end]))    
    Xs = X[model.subset,:]; (eigvals,eigvec) = eig_AFEm(Xs, k)
    # generate model from the parameters
    model = model_from_parameters(model,x0)
    # perform cross-validation by a generic and parallelizable function
    gen_cross_validate(X, Y, folds=model.num_cv_folds) do Xtr, Ytr, Xval, Yval
        # perform Automatic Feature Extraction by Nystrom method 
        features_train = AFEm(eigvals,eigvec,Xs,k,Xtr)
        features_valid = AFEm(eigvals,eigvec,Xs,k,Xval)        
        # run algorithm        
        (model.w, model.b) = run_algorithm(features_train,Ytr,model)
        validation_criteria(model,features_valid,Yval)
    end
end