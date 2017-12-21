# 
# Software Lab for Advanced Machine Learning with Stochastic Algorithms
# Copyright (c) 2015 Vilen Jumutc, KU Leuven, ESAT-STADIUS 
# License & help @ https://github.com/jumutc/SALSA.jl
# Documentation @ http://salsajl.readthedocs.org
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#

function tune_algorithm_AFEm(X, Y, model::SALSAModel)
    # Renyi entropy calculations
    k = kernel_from_data_model(model.kernel,X)    
    rp = ceil(sqrt(size(X,1)*model.subset_size))
    num_k = length(fieldnames(model.kernel))
    X_subset = sub(X,entropy_subset(X,k,rp),:)
    
    cost_fun = x0 -> cross_validate_algorithm_AEFm(x0,X,Y,model,num_k,X_subset)
    par = run_global_opt(model,cost_fun,model.global_opt,(size(Y,2)*5+num_k,5))
    
    # set the output model mode correctly
    pars = num_k > 0 ? exp(par[end-num_k+1:end]) : []
    model.output.mode = NONLINEAR(pars,X_subset)
    # return model and the parameters
    model, par
end

function cross_validate_algorithm_AEFm(x0, X, Y, model, num_k, X_subset)
    pars = num_k > 0 ? exp(x0[end-num_k+1:end]) : []
    kernel = kernel_from_parameters(model.kernel,pars)
    (eigvals,eigvec) = eig_AFEm(X_subset, kernel)
    
    # perform cross-validation by a generic and parallelizable function
    gen_cross_validate(size(Y,1), model) do train_idx, val_idx
        Xtr, Ytr   = view(X,train_idx,:), view(Y,train_idx,:)
        Xval, Yval = view(X,val_idx,:), view(Y,val_idx,:)
        # perform Automatic Feature Extraction by Nystrom approximation 
        features_train = AFEm(eigvals,eigvec,X_subset,kernel,Xtr)
        features_valid = AFEm(eigvals,eigvec,X_subset,kernel,Xval)        
        # run & validate algorithm        
        (model.output.w, model.output.b) = run_with_params(features_train,Ytr,model,x0)
        validation_criterion(model,features_valid,Yval)
    end
end