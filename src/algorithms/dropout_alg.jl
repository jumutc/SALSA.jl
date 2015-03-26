export dropout_alg 

function dropout_alg(dfunc::Function, X, Y, λ::Float64, k::Int, max_iter::Int, tolerance::Float64, online_pass=false, train_idx = [])
    # Internal function for a simple Dropout Pegasos routine
    #
    # Copyright (c) 2015, KU Leuven-ESAT-STADIUS, License & help @
    # http://www.esat.kuleuven.be/sista/pegasoslab

    N = size(X,1)
    d = size(X,2) + 1
    check = issparse(X)
    
    if ~check
        w = rand(d)
        rw = ones(d)/d 
        A = [X'; ones(1,N)]
    else 
        total = length(X.nzval)
        w = sprand(d,1,total/(N*d))
        A = [X'; sparse(ones(1,N))]
        f_sample = (p) -> rand(Bernoulli(p^2/(1+p^2))) 
    end

    if ~isempty(train_idx)
        space = train_idx
        N = size(space,1)
    else
        space = 1:1:N
    end

    if online_pass
        max_iter = N
        smpl = (t,k) -> t
    else
        pd = Categorical(N)
        smpl = (t,k) -> rand(pd,k)
    end

    for t=1:max_iter 
        idx = space[smpl(t,k)]
        w_prev = w
        
        yt = Y[idx]
        At = A[:,idx]
       
        # do not perform transpose(::SparceMatrixCSC) and other operations 
        # because Garbage Collection performs realy badly in the tight loops
        eval = map(i->sum(At[:,i].*w),1:1:k).*yt
        idx1 = find(eval .< 1)

        # define samples
        if ~check
            prob = map(p -> Bernoulli(1-p),rw)
            dropout = map(rand, prob) 
        else
            bern_vars = map(f_sample,w.nzval)
            dropout = SparseMatrixCSC(d,1,w.colptr,w.rowval,bern_vars)
            dropout = reduce_sparsevec(dropout,find(bern_vars))
        end

        # do a gradient descent step
        grad = dfunc(At,yt,idx1)
        w = w - (dropout.*w)./t
        w = w + (λ/(t*k))*grad

        if ~check
            rw = 1 ./ (1 + w.^2)
        end

        # check the stopping criteria w.r.t. Tolerance, check, online_pass
        if ~online_pass && ~check && vecnorm(w - w_prev) < tolerance
            break
        end
    end

    w[1:end-1], w[end]
end