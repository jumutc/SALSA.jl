export stochastic_rk_means

# extensive set of multiplicated aliases for different supporting algorithms 
run_algorithm(::Type{PEGASOS}, X, Y, dfunc::Function, alg_params::Vector, k::Int, max_iter::Int, tolerance::Float64, online_pass, train_idx) = 
			  pegasos_alg(dfunc, X, Y, alg_params..., k, max_iter, tolerance, online_pass, train_idx) 
run_algorithm(::Type{SGD}, X, Y, dfunc::Function, alg_params::Vector, k::Int, max_iter::Int, tolerance::Float64, online_pass, train_idx) = 
			  sgd_alg(dfunc, X, Y, alg_params..., k, max_iter, tolerance, online_pass, train_idx) 
run_algorithm(::Type{L1RDA}, X, Y, dfunc::Function, alg_params::Vector, k::Int, max_iter::Int, tolerance::Float64, online_pass, train_idx) = 
			  l1rda_alg(dfunc, X, Y, alg_params..., k, max_iter, tolerance, online_pass, train_idx) 
run_algorithm(::Type{R_L1RDA}, X, Y, dfunc::Function, alg_params::Vector, k::Int, max_iter::Int, tolerance::Float64, online_pass, train_idx) = 
			  reweighted_l1rda_alg(dfunc, X, Y, alg_params..., k, max_iter, tolerance, online_pass, train_idx) 
run_algorithm(::Type{R_L2RDA}, X, Y, dfunc::Function, alg_params::Vector, k::Int, max_iter::Int, tolerance::Float64, online_pass, train_idx) = 
			  reweighted_l2rda_alg(dfunc, X, Y, alg_params..., k, max_iter, tolerance, online_pass, train_idx) 
run_algorithm(::Type{DROP_OUT}, X, Y, dfunc::Function, alg_params::Vector, k::Int, max_iter::Int, tolerance::Float64, online_pass, train_idx) = 
			  dropout_alg(dfunc, X, Y, alg_params..., k, max_iter, tolerance, online_pass, train_idx)
run_algorithm(::Type{ADA_L1RDA}, X, Y, dfunc::Function, alg_params::Vector, k::Int, max_iter::Int, tolerance::Float64, online_pass, train_idx) = 
			  adaptive_l1rda_alg(dfunc, X, Y, alg_params..., k, max_iter, tolerance, online_pass, train_idx)

# core algorithmic part
function stochastic_rk_means{A <: Algorithm}(X, rk_means::RK_MEANS{A}, alg_params::Vector, k::Int, max_iter::Int, 
											 tolerance::Float64, online_pass=0, train_idx=[])
	# Internal function for a simple Stochastic Regularized K-means
    #
    # Copyright (c) 2015, KU Leuven-ESAT-STADIUS, License & help @
    # http://www.esat.kuleuven.be/stadius/ADB/jumutc/softwareSALSA.php

	N, d  = size(X)
    check = issparse(X)
    dfunc = loss_derivative(RK_MEANS)
    
    if ~check
        w = rand(d,rk_means.k_clusters)
    else 
        w = sprand(d,rk_means.k_clusters,length(X.nzval)/(N*d))
    end

    if isempty(train_idx)
    	train_idx = 1:1:N
    end

    failed_mapping = false; t = 1; Y = ones(N)

    while true
    	dists = pairwise(rk_means.metric, convert(Array, sub(X,train_idx,:))', w)
    	(x,y) = findn(dists .== minimum(dists,2))
    	mappings = zeros(length(train_idx))
    	mappings[x] = y

    	if ~failed_mapping &&  t > rk_means.max_iter
    		break 
    	end

    	if t > rk_means.max_iter*10
    		break # exit if everything fails
    	end
    	
    	result = @parallel (hcat) for cluster_id in unique(mappings)
    		cluster_idx = find(mappings .== cluster_id)
    		run_algorithm(rk_means.support_alg,X,Y,dfunc,alg_params,k,max_iter,tolerance,online_pass,cluster_idx)[1]
    	end
    	
    	# assign and check the result of parallel execution
    	if all(result .== 0) || all(isnan(result))
    		failed_mapping = true
    		w = rand(d,rk_means.k_clusters)
    	elseif size(result,2) != size(w,2)
    		failed_mapping = true
    		diff = size(w,2) - size(result,2) 
    		w = [result rand(d,diff)]
    	else
    		failed_mapping = false
    		w = result 
    	end

    	t += 1
    end    	

    w
end 