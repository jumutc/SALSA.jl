function gen_cross_validate(evalfun::Function, n::Int, model::SALSAModel)
	indices = get(model.cv_gen, Kfold(n,nfolds())) 
    @parallel (+) for train_idx in collect(indices)
		val_idx = setdiff(1:n, train_idx)
        evalfun(train_idx, val_idx)/nfolds()
    end
end