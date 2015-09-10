function gen_cross_validate(evalfun::Function, n::Int, model::SALSAModel)
	n_f = nfolds()
	if isdefined(model.output,:cv_folds) && (n == model.output.cv_n) && (n_f == model.output.cv_n_f)
		folds = model.output.cv_folds #use stored values
	else
		indices = get(model.cv_gen, Kfold(n, n_f)) #compute folds
		folds = collect(indices)
		n_f = length(folds)
		
		model.output.cv_folds = folds #store for later
		model.output.cv_n_f = n_f
		model.output.cv_n = n
	end
	
    @parallel (+) for train_idx in folds
		val_idx = setdiff(1:n, train_idx)
        evalfun(train_idx, val_idx)/n_f
    end
end