#
# Software Lab for Advanced Machine Learning with Stochastic Algorithms
# Copyright (c) 2015 Vilen Jumutc, KU Leuven, ESAT-STADIUS
# License & help @ https://github.com/jumutc/SALSA.jl
# Documentation @ http://salsajl.readthedocs.org
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#

function dropout_alg(dfunc::Function, X, Y, λ::Float64, k::Int, max_iter::Int, tolerance::Float64, online_pass=0, train_idx=[])
    # Internal function for a simple Dropout Pegasos routine
    #
    # Copyright (c) 2015, KU Leuven-ESAT-STADIUS, License & help @
    # http://www.esat.kuleuven.be/stadius/ADB/jumutc/softwareSALSA.php

    N = size(X,1)
    d = size(X,2) + 1
    check = issparse(X)

    if ~check
        w = rand(d)
        rw = ones(d)/d
        sub_arr = (I) -> append_ones(sub(X,I,:),k)
    else
        total = length(X.nzval)
        w = sparsevec(sprand(d,1,total/(N*d)))
        f_sample = (p) -> isnan(p^2/(1+p^2)) ?
              rand(Bernoulli(0)) : rand(Bernoulli(p^2/(1+p^2)))
        X = X'; sub_arr = (I) -> append_ones(X[:,I],k)
    end

    space, N = fix_space(train_idx,N)
    smpl = fix_sampling(online_pass,N)
    max_iter = fix_iter(online_pass,N,max_iter)

    for t=1:max_iter
        idx = space[smpl(t,k)]
        w_prev = w

        yt = Y[idx]
        At = sub_arr(idx)
        grad = dfunc(At,yt,w)

        # define samples
        if ~check
            prob = map(p -> Bernoulli(1-p),rw)
            dropout = map(rand, prob)
        else
            bern_vars = map(f_sample,w.nzval)
            dropout = SparseVector(d,w.nzind,bern_vars)
            dropout = reduce_sparsevec(dropout,find(bern_vars))
            grad = sparsevec(grad)
        end

        # do a gradient descent step
        w = w - (1/t)*(dropout.*w)
        w = w - (1/(λ*t*k))*grad

        if ~check
            rw = 1 ./ (1 + w.^2)
        end

        # check the stopping criterion w.r.t. Tolerance, check, online_pass
        if online_pass == 0 && ~check && vecnorm(w - w_prev) < tolerance
            break
        end
    end

    w[1:end-1], w[end]
end
