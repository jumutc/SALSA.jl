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