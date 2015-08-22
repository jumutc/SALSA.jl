function gen_cross_validate(evalfun::Function, X, Y, model::SALSAModel)
	indices = get(model.cv_gen, Kfold(length(Y),nfolds()))
    @parallel (+) for train_idx in collect(indices)
		val_idx = setdiff(1:length(Y), train_idx)
        evalfun(sub(X,train_idx,:), Y[train_idx,:], sub(X,val_idx,:), Y[val_idx,:])/nfolds()
    end
end

function gen_cross_validate(evalfun::Function, n::Int, model::SALSAModel)
	indices = get(model.cv_gen, Kfold(n,nfolds())) 
    @parallel (+) for train_idx in collect(indices)
		val_idx = setdiff(1:n, train_idx)
        evalfun(train_idx, val_idx)/nfolds()
    end
end