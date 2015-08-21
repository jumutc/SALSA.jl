# External function for a complete stochastic learning routine with cross-validation
	#
	# Copyright (c) 2015, KU Leuven-ESAT-STADIUS, License & help @
	# http://www.esat.kuleuven.be/stadius/ADB/software.php
	#
	#    model = salsa(X,Y,model,Xtest) runs Pegasos, RDA... stochastic
	#    algorithms suited for large-scale binary data classification
	#
	# Based on:
	# 
    # [1] S. Shalev-Shwartz, Y. Singer, N. Srebro, Pegasos: Primal Estimated sub-GrAdient SOlver for SVM, 
    # in: Proceedings of the 24th international conference on Machine learning, ICML ’07, New York, NY, USA, 2007, pp. 807–814.

    # [2] L. Xiao, Dual averaging methods for regularized stochastic learning and online optimization, 
    # J. Mach. Learn. Res. 11 (2010) 2543–2596.

    # [3] J. Duchi, E. Hazan, Y. Singer, Adaptive subgradient methods for online learning and stochastic optimization, 
    # J. Mach. Learn. Res. 12 (2011) 2121–2159.

    # [4] V. Jumutc, J. A. K. Suykens, Reweighted l1 dual averaging approach for sparse stochastic learning, 
    # in: 22th European Symposium on Artificial Neural Networks, ESANN 2014, Bruges, Belgium, April 23-25, 2014.

    # [5] V. Jumutc, J. A. K. Suykens, Reweighted l2 -regularized dual averaging approach for highly sparse stochastic learning, 
    # in: Advances in Neural Networks - 11th International Symposium on Neural Networks, ISNN 2014, 
    # Hong Kong and Macao, China, November 28 – December 1, 2014, pp. 232–242.
	
function salsa(X, Y, model::SALSAModel, Xtest)
	if ~isdefined(model,:output)
		model.output = OutputModel{model.mode}()
	end

    if model.normalized && isempty(Xtest) && typeof(X) <: Array
	    (X, model.output.X_mean, model.output.X_std) = mapstd(X)
	elseif model.normalized && typeof(X) <: Array
	    (X, model.output.X_mean, model.output.X_std) = mapstd(X)
	    Xtest = mapstd(Xtest,model.output.X_mean,model.output.X_std)
	end

	if model.mode == LINEAR
	    model = tune_algorithm(X,Y,model)
	    (model.output.w, model.output.b) = run_algorithm(X,Y,model)
	else
	    model = tune_algorithm_AFEm(X,Y,model) 
	    # find actual Nystrom-approximated feature map and run Pegasos
	    k = kernel_from_parameters(model.kernel,model.output.mode.k_params)
	    features = AFEm(model.output.mode.X_subset,k,X)
	    (model.output.w, model.output.b) = run_algorithm(features,Y,model)	    
	end

	if !isempty(Xtest)
	    model.output.Ytest = predict(model.validation_criteria,model,Xtest)
	end

	model
end