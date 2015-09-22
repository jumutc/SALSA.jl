# extensive set of multiplicated aliases for different supporting algorithms 
run_algorithm(::Type{PEGASOS}, X, Y, dfunc::Function, alg_params::Vector, k::Int, max_iter::Int, tolerance::Float64, online_pass, train_idx) = 
			  pegasos_alg(dfunc, X, Y, alg_params..., k, max_iter, tolerance, online_pass, train_idx) 
run_algorithm(::Type{SIMPLE_SGD}, X, Y, dfunc::Function, alg_params::Vector, k::Int, max_iter::Int, tolerance::Float64, online_pass, train_idx) = 
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

At_mul_B!(C::Array{Float64,2}, A::SparseMatrixCSC, B::SparseMatrixCSC) = begin C[:,:] = At_mul_B(A,B) end
At_mul_B!(C::Array{Float64,2}, A::SparseMatrixCSC, B::Array{Float64,2}) = At_mul_B!(C,full(A),B)
At_mul_B!(C::Array{Float64,2}, A::Array{Float64,2}, B::SparseMatrixCSC) = At_mul_B!(C,A,full(B))

# core algorithmic part
function stochastic_rk_means(X, rk_means::RK_MEANS, alg_params::Vector, k::Int, max_iter::Int, 
							 tolerance::Float64, online_pass=0, train_idx=[])
	# Internal function for a simple Stochastic Regularized K-means
    #
    # Copyright (c) 2015, KU Leuven-ESAT-STADIUS, License & help @
    # http://www.esat.kuleuven.be/stadius/ADB/jumutc/softwareSALSA.php

	N, d  = size(X)
    check = issparse(X)
    dfunc = loss_derivative(rk_means)
    
    if ~check
        w = rand(d,rk_means.k_clusters)
    else 
        w = sprand(d,rk_means.k_clusters,length(X.nzval)/(N*d))
    end

    if isempty(train_idx)
    	X_ = sub(X,:,:)'
    else
        X_ = sub(X,train_idx,:)'
    end

    failed_mapping = false; t = 1; Y = ones(N)

    while true
        dists = pairwise(rk_means.metric, X_, w)
        (x,y) = findn(dists .== minimum(dists,2))
        mappings = zeros(size(X_,2))
        mappings[x] = y
        mappings

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
    	elseif size(result) != size(w)
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