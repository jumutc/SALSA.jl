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

function At_mul_B!(C::Array{Float64,2}, A::SparseMatrixCSC, B::SparseMatrixCSC)
    assert(size(A,1) == size(B,1))
    C[:,:] = At_mul_B(A,B)
end

# core algorithmic part
function stochastic_rk_means{A <: Algorithm}(X, rk_means::RK_MEANS{A}, alg_params::Vector, k::Int, max_iter::Int, 
											 tolerance::Float64, online_pass=0, train_idx=[]; num_blocks=25)
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
    	train_idx = 1:1:N
    end

    total_size = length(train_idx)
    block_size = round(Int,total_size/num_blocks)
    failed_mapping = false; t = 1; Y = ones(N)

    while true
        mappings = @parallel (vcat) for block=1:num_blocks
            to_ind = block_size*block
            to_ind = block == num_blocks ? total_size : to_ind
            idx   = train_idx[num_blocks*(block-1)+1:to_ind]
            dists = pairwise(rk_means.metric, sub(X,idx,:)', w)
            (x,y) = findn(dists .== minimum(dists,2))
            assignments = zeros(length(idx))
            assignments[x] = y
            assignments
        end

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