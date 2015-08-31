function gen_cross_validate(evalfun::Function, n::Int, model::SALSAModel)
	indices = get(model.cv_gen, Kfold(n,nfolds()))
	folds = collect(indices); n_f = length(folds)
    @parallel (+) for train_idx in folds
		val_idx = setdiff(1:n, train_idx)
        evalfun(train_idx, val_idx)/n_f
    end
end