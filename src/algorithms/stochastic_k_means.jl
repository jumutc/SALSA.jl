export stochastic_k_means

# extensive set of multiplicated aliases for different supporting algorithms 
run_algorithm(::Type{PEGASOS}, X, Y, dfunc::Function, alg_params::Vector, k::Int, max_iter::Int, tolerance::Float64, online_pass, train_idx) = 
			  pegasos_alg(dfunc, X, Y, alg_params..., k, max_iter, tolerance, online_pass, train_idx) 
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
function stochastic_k_means{A <: Algorithm}(dfunc::Function, X, k_means::K_MEANS{A}, alg_params::Vector, 
							k::Int, max_iter::Int, tolerance::Float64, online_pass=false, train_idx=[])
	# Internal function for a simple stochastic k-means routine
    #
    # Copyright (c) 2015, KU Leuven-ESAT-STADIUS, License & help @
    # http://www.esat.kuleuven.be/stadius/ADB/jumutc/softwareSALSA.php

	N, d  = size(X)
    check = issparse(X)
    is_converged = false 
    
    if ~check
    	b = zeros(1,k_means.k_clusters)
        w = rand(d,k_means.k_clusters)
    else 
    	b = spzeros(1,k_means.k_clusters)
        w = sprand(d,k_means.k_clusters,length(X.nzval)/(N*d))
    end

    if isempty(train_idx)
    	train_idx = 1:1:N
    end

    cluster_mappings = zeros(length(train_idx)); Y = zeros(N)

    for t=1:k_means.max_iter
    	eval = abs(X[train_idx,:]*w+repmat(b,N,1))
    	mappings = findn(eval.==minimum(eval,2))[2]
    	print(size(eval))

    	if all(cluster_mappings.==mappings)
    		break # check cluster mappings has not changed and exit
    	else
	    	result = @parallel (hcat) for cluster_id in unique(mappings)
	    		cluster_idx = train_idx[find(mappings.==cluster_id)]
	    		(w,b) = run_algorithm(k_means.support_alg,X,Y,dfunc,alg_params,k,max_iter,tolerance,online_pass,cluster_idx)
	    		[w;b]
	    	end
	    	# assign the result of parallel execution
	    	w = result[1:end-1,:]; b = result[end,:]
	    	cluster_mappings = mappings
    	end 
    end    	

    w, b
end 