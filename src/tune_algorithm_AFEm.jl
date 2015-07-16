function tune_algorithm_AFEm(X, Y, model::SALSAModel)
    # Renyi entropy calculations
    k = kernel_from_data_model(model.kernel,X)    
    rp = ceil(sqrt(size(X,1)*model.subset_size))
    num_k = length(fieldnames(model.kernel))
    X_subset = sub(X,entropy_subset(X,k,rp),:)
    
    cost_fun = x0 -> cross_validate_algorithm_AEFm(x0,X,Y,model,num_k,X_subset)
    par = run_global_opt(model,cost_fun,model.global_opt,n_params=5+num_k)
    
    # set the output model mode correctly
    pars = num_k > 0 ? exp(par[end-num_k+1:end]) : []
    model.output.mode = NONLINEAR(pars,X_subset[:,:])
    # generate model from the parameters
    model_from_parameters(model,par)
end

function cross_validate_algorithm_AEFm(x0, X, Y, model, num_k, X_subset)
    pars = num_k > 0 ? exp(x0[end-num_k+1:end]) : []
    k = kernel_from_parameters(model.kernel,pars)    
    (eigvals,eigvec) = eig_AFEm(X_subset, k)
    # generate model from the parameters
    model = model_from_parameters(model,x0)
    # perform Kfold cross-validation by a generic and parallelizable function
    gen_cross_validate(X, Y, model) do Xtr, Ytr, Xval, Yval
        # perform Automatic Feature Extraction by Nystrom method 
        features_train = AFEm(eigvals,eigvec,X_subset,k,Xtr)
        features_valid = AFEm(eigvals,eigvec,X_subset,k,Xval)        
        # run algorithm        
        (model.output.w, model.output.b) = run_algorithm(features_train,Ytr,model)
        validation_criteria(model,features_valid,Yval)
    end
end