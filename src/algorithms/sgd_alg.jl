function sgd_alg(dfunc::Function, X, Y, λ::Float64, k::Int, max_iter::Int, tolerance::Float64, online_pass=0, train_idx=[])
    # Internal function for a simple SGD routine for λ-strongly convex functions
    #
    # Copyright (c) 2015, KU Leuven-ESAT-STADIUS, License & help @
    # http://www.esat.kuleuven.be/stadius/ADB/jumutc/softwareSALSA.php

    N = size(X,1)
    d = size(X,2) + 1
    check = issparse(X) 
    
    if ~check
        w = rand(d)
        sub_arr = (I) -> [sub(X,I,:) ones(k,1)]'
    else 
        total = length(X.nzval)
        w = sprand(d,1,total/(N*d))
        X = [X'; sparse(ones(1,N))]
        sub_arr = (I) -> X[:,I]
    end

    if ~isempty(train_idx)
        space = train_idx
        N = size(space,1)
    else
        space = 1:1:N
    end

    if online_pass > 0
        max_iter = N*online_pass
        smpl = (t,k) -> begin
            s = t % N 
            s > 0 ? s : N
        end
    else
        pd = Categorical(N)
        smpl = (t,k) -> rand(pd,k)
    end

    for t=1:max_iter 
        idx = space[smpl(t,k)]
        w_prev = w
        
        yt = Y[idx]
        At = sub_arr(idx)
       
        # do a gradient descent step
        η_t = 1/(λ*t)
        w = (1 - λ*η_t).*w
        w = w - (η_t/k).*dfunc(At,yt,w_prev)
        
        # check the stopping criteria w.r.t. Tolerance, check, online_pass
        if online_pass == 0 && ~check && vecnorm(w - w_prev) < tolerance
            break
        end
    end

    w[1:end-1,:], w[end,:]
end