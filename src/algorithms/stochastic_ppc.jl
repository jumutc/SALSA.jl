export stochastic_ppc

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
function stochastic_ppc{A <: Algorithm}(dfunc::Function, X, ppc::PPC{A}, alg_params::Vector, 
							k::Int, max_iter::Int, tolerance::Float64, online_pass=false, train_idx=[])
	# Internal function for a simple stochastic Proximal Plane Clustering
    #
    # Copyright (c) 2015, KU Leuven-ESAT-STADIUS, License & help @
    # http://www.esat.kuleuven.be/stadius/ADB/jumutc/softwareSALSA.php

	N, d  = size(X)
    check = issparse(X)
    is_converged = false 
    
    if ~check
    	b = zeros(1,ppc.k_clusters)
        w = rand(d,ppc.k_clusters)
    else 
    	b = spzeros(1,ppc.k_clusters)
        w = sprand(d,ppc.k_clusters,length(X.nzval)/(N*d))
    end

    if isempty(train_idx)
    	train_idx = 1:1:N
    end

    cluster_mappings = zeros(length(train_idx)) 
    failed_mapping = false
    Y = zeros(N); t = 1

    while true
    	eval = abs(X[train_idx,:]*w + repmat(b,N,1))
    	mappings = findn(eval .== minimum(eval,2))[2]
    	mappings_not_changed = all(cluster_mappings .== mappings)

    	if ~failed_mapping && (mappings_not_changed || t > ppc.max_iter) 
    		break # check cluster mappings has not changed and exit
    	end

    	if t > ppc.max_iter*10
    		break # exit if everything fails
    	end
    	
    	result = @parallel (hcat) for cluster_id in unique(mappings)
    		cluster_idx = train_idx[find(mappings .== cluster_id)]
    		(w,b) = run_algorithm(ppc.support_alg,X,Y,dfunc,alg_params,k,max_iter,tolerance,online_pass,cluster_idx)
    		[w;b]
    	end
    	
    	# assign and check the result of parallel execution
    	if all(result .== 0) || all(isnan(result))
    		failed_mapping = true
    		w = rand(d,ppc.k_clusters)
    		b = zeros(1,ppc.k_clusters)
    	elseif size(result,2) != size(w,2)
    		failed_mapping = true
    		diff = size(w,2) - size(result,2) 
    		w = [result[1:end-1,:] rand(d,diff)]
    		b = [result[end,:] rand(1,diff)]
    	else
    		failed_mapping = false
    		w = result[1:end-1,:]; b = result[end,:]
    	end

    	cluster_mappings = mappings; t += 1
    end    	

    w, b
end 