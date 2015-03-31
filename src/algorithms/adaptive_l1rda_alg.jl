export adaptive_l1rda_alg 

function adaptive_l1rda_alg(dfunc::Function, X, Y, λ::Float64, γ::Float64, ρ::Float64, 
                   k::Int, max_iter::Int, tolerance::Float64, online_pass=false, train_idx = [])

    # Internal function for a simple Adaptive l1-RDA routine
    #
    # Copyright (c) 2015, KU Leuven-ESAT-STADIUS, License & help @
    # http://www.esat.kuleuven.be/sista/pegasoslab

    N = size(X,1)
    d = size(X,2) + 1
    check = ~issparse(X) 
    
    if check
        w = zeros(d,1)
        g = zeros(d,1)
        h = zeros(d,1)
        A = [X'; ones(1,N)]
    else 
        w = spzeros(d,1)
        g = spzeros(d,1)
        h = spzeros(d,1)
        A = [X'; sparse(ones(1,N))]
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

        # calculate dual average: (cumulative) gradient
        g_new = dfunc(At,yt,eval)
        g = ((t-1)/t).*g + (1/(t)).*g_new
  
        # find a close form solution
        if check
            h = t>1 ? h + g_new.^2 : ρ .+ g_new.^2 
            w = -(t*γ./sqrt(h)).*(g - λ.*sign(g))
            w[abs(g).<=λ] = 0
        else
            # do not perform sparse(...) and filter and map over SparceMatrixCSC
            # because Garbage Collection performs realy badly in the tight loops
            h = t>1 ? h + g_new.^2 : ρ.*spones(g_new) + g_new.^2 
            h_sq = SparseMatrixCSC(d,1,h.colptr,h.rowval,-(t*γ)./sqrt(h.nzval))
            gs = SparseMatrixCSC(d,1,g.colptr,g.rowval,sign(g.nzval))
            w = h_sq.*(g - λ.*gs); ind = abs(g.nzval) .> λ
            w = reduce_sparsevec(w,find(ind)) 
        end

        # check the stopping criteria w.r.t. Tolerance, check, online_pass
        if ~online_pass && check && vecnorm(w - w_prev) < tolerance
            break
        end
    end

    w[1:end-1,:], w[end,:]
end