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
	# sanity checks of input data
	assert(size(Y,1) == size(X,1))

	if ~isdefined(model,:output)
		model.output = OutputModel{model.mode}()
	end

    if model.normalized && typeof(X) <: Array && isempty(Xtest) 
	    (X, model.output.X_mean, model.output.X_std) = mapstd(X)
	elseif model.normalized && typeof(X) <: Array
	    (X, model.output.X_mean, model.output.X_std) = mapstd(X)
	    Xtest = mapstd(Xtest,model.output.X_mean,model.output.X_std)
	end

	# perform aka OneHotEncoding if needed (-1/1 encoding is not given)
	if model.process_labels && length(setdiff([-1,1],unique(Y))) > 0
		Y_ = sort(unique(Y)); k = length(Y_)
		encoding = -ones(size(X,1),k)
		for y in zip(1:size(X,1),indexin(Y,Y_))
			encoding[y[1],y[2]] = 1
		end
		Y = encoding # re-define Y input
	end

	(model.output.w, model.output.b) = salsa(X,Y,model)

	if size(Y,2) > 1 && !isempty(Xtest) && typeof(model.algorithm) != RK_MEANS  
		# multi-class case (One vs. All)
		model.output.Ytest = membership(predict_latent(model,Xtest))''
	elseif !isempty(Xtest) # binary or regression case
		model.output.Ytest = predict(model,Xtest)
	end
	
	model
end

function salsa(X, Y, model::SALSAModel)
	if model.mode == LINEAR
	    model, pars = tune_algorithm(X,Y,model)
	    run_with_params(X,Y,model,pars)
	else
	    model, pars = tune_algorithm_AFEm(X,Y,model) 
	    # find actual Nystrom-approximated feature map and run Pegasos
	    kernel = kernel_from_parameters(model.kernel,model.output.mode.k_params)
	    features_train = AFEm(model.output.mode.X_subset,kernel,X)
	    run_with_params(features_train,Y,model,pars)
	end
end

# extensive set of multiplicated aliases for different algorithms and models /// dense matrices
salsa{L <: Loss, A <: Algorithm, M <: Mode, N1 <: Number, N2 <: Number}(mode::Type{M}, alg::Type{A}, loss::Type{L}, X::Array{N1,2}, Y::Array{N2,1}, Xtest::Array{N1,2}) = salsa(X,Y,SALSAModel(mode,alg(),loss),Xtest)
salsa{L <: Loss, A <: Algorithm, M <: Mode, N1 <: Number, N2 <: Number}(mode::Type{M}, alg::Type{A}, loss::Type{L}, X::Array{N1,2}, Y::Array{N2,2}, Xtest::Array{N1,2}) = salsa(X,Y,SALSAModel(mode,alg(),loss),Xtest)
salsa{A <: Algorithm, N1 <: Number, N2 <: Number}(alg::Type{A}, X::Array{N1,2}, Y::Array{N2,1}, Xtest::Array{N1,2}) = salsa(X,Y,SALSAModel(LINEAR,alg(),HINGE),Xtest)
salsa{A <: Algorithm, N1 <: Number, N2 <: Number}(alg::Type{A}, X::Array{N1,2}, Y::Array{N2,2}, Xtest::Array{N1,2}) = salsa(X,Y,SALSAModel(LINEAR,alg(),HINGE),Xtest)
salsa{N1 <: Number, N2 <: Number}(X::Array{N1,2}, Y::Array{N2,1}, Xtest::Array{N1,2}) = salsa(LINEAR,PEGASOS,HINGE,X,Y,Xtest)
salsa{N1 <: Number, N2 <: Number}(X::Array{N1,2}, Y::Array{N2,2}, Xtest::Array{N1,2}) = salsa(LINEAR,PEGASOS,HINGE,X,Y,Xtest)
salsa{N1 <: Number, N2 <: Number}(X::Array{N1,2}, Y::Array{N2,1}) = salsa(LINEAR,PEGASOS,HINGE,X,Y,Array{Int64}(0,0))
salsa{N1 <: Number, N2 <: Number}(X::Array{N1,2}, Y::Array{N2,2}) = salsa(LINEAR,PEGASOS,HINGE,X,Y,Array{Int64}(0,0))
# extensive set of multiplicated aliases for different algorithms and models /// sparse matrices
salsa{L <: Loss, A <: Algorithm, M <: Mode, N <: Number}(mode::Type{M}, alg::Type{A}, loss::Type{L}, X::SparseMatrixCSC, Y::Array{N,1}, Xtest::SparseMatrixCSC) = salsa(X,Y,SALSAModel(mode,alg(),loss),Xtest)
salsa{L <: Loss, A <: Algorithm, M <: Mode, N <: Number}(mode::Type{M}, alg::Type{A}, loss::Type{L}, X::SparseMatrixCSC, Y::Array{N,2}, Xtest::SparseMatrixCSC) = salsa(X,Y,SALSAModel(mode,alg(),loss),Xtest)
salsa{A <: Algorithm, N <: Number}(alg::Type{A}, X::SparseMatrixCSC, Y::Array{N,1}, Xtest::SparseMatrixCSC) = salsa(X,Y,SALSAModel(LINEAR,alg(),HINGE),Xtest)
salsa{A <: Algorithm, N <: Number}(alg::Type{A}, X::SparseMatrixCSC, Y::Array{N,2}, Xtest::SparseMatrixCSC) = salsa(X,Y,SALSAModel(LINEAR,alg(),HINGE),Xtest)
salsa{N <: Number}(X::SparseMatrixCSC, Y::Array{N,1}, Xtest::SparseMatrixCSC) = salsa(LINEAR,PEGASOS,HINGE,X,Y,Xtest)
salsa{N <: Number}(X::SparseMatrixCSC, Y::Array{N,2}, Xtest::SparseMatrixCSC) = salsa(LINEAR,PEGASOS,HINGE,X,Y,Xtest)
salsa{N <: Number}(X::SparseMatrixCSC, Y::Array{N,1}) = salsa(LINEAR,PEGASOS,HINGE,X,Y,sparse([]))
salsa{N <: Number}(X::SparseMatrixCSC, Y::Array{N,2}) = salsa(LINEAR,PEGASOS,HINGE,X,Y,sparse([]))
# extensive set of multiplicated aliases for different algorithms and models /// DelimitedFile
salsa{L <: Loss, A <: Algorithm, M <: Mode, N <: Number}(mode::Type{M}, alg::Type{A}, loss::Type{L}, X::DelimitedFile, Y::Array{N,1}, Xtest::DelimitedFile) = salsa(X,Y,SALSAModel(mode,alg(),loss),Xtest)
salsa{L <: Loss, A <: Algorithm, M <: Mode, N <: Number}(mode::Type{M}, alg::Type{A}, loss::Type{L}, X::DelimitedFile, Y::Array{N,2}, Xtest::DelimitedFile) = salsa(X,Y,SALSAModel(mode,alg(),loss),Xtest)
salsa{N <: Number}(X::DelimitedFile, Y::Array{N,1}, Xtest::DelimitedFile) = salsa(LINEAR,PEGASOS,HINGE,X,Y,Xtest)
salsa{N <: Number}(X::DelimitedFile, Y::Array{N,2}, Xtest::DelimitedFile) = salsa(LINEAR,PEGASOS,HINGE,X,Y,Xtest)