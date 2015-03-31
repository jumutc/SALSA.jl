export reweighted_l2rda_alg 

function reweighted_l2rda_alg(dfunc::Function, X, Y, λ::Float64, ɛ::Float64, varɛ::Float64, 
                              k::Int, max_iter::Int, tolerance::Float64, online_pass=false, train_idx = [])

    # Internal function for a simple Reweighted l2-RDA routine
    #
    # Copyright (c) 2015, KU Leuven-ESAT-STADIUS, License & help @
    # http://www.esat.kuleuven.be/sista/pegasoslab

    N = size(X,1)
    d = size(X,2) + 1
    check = ~issparse(X)
    rw = ones(d)
    
    if check
        w = zeros(d)
        g = zeros(d)
        A = [X'; ones(1,N)]
    else      
        w = spzeros(d,1)
        g = spzeros(d,1)
        A = [X'; sparse(ones(1,N))]
        fg = i -> -1./(λ .+ rw[i])
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

        # calculate dual average: gradient
        g = ((t-1)/t).*g + (1/(t)).*dfunc(At,yt,eval)
        
        # find a close form solution
        # update re-weighting vector
        if check     
            w = -(1./(λ .+ rw)).*g       
            rw = 1 ./ (w.^2 .+ ɛ)
        else 
            # do not perform sparse(...) and filter and map over SparceMatrixCSC
            # because Garbage Collection performs realy badly in the tight loops
            w = SparseMatrixCSC(d,1,g.colptr,g.rowval,fg(g.rowval)).*g
            rw[g.rowval] = 1./(ɛ .+ w.nzval.^2)
        end
        
        # check the stopping criteria w.r.t. Tolerance, check, online_pass
        if ~online_pass && check && vecnorm(w - w_prev) < tolerance
            break
        end
    end

    # truncate solution
    if check
        w[abs(w).<=varɛ] = 0
    else
        ind = abs(w.nzval) .> varɛ
        w = reduce_sparsevec(w,find(ind))
    end

    w[1:end-1], w[end]
end