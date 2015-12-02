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

function pegasos_alg(dfunc::Function, X, Y, λ::Float64, k::Int, max_iter::Int, tolerance::Float64, online_pass=0, train_idx=[])
    # Internal function for a simple Pegasos routine
    #
    # Copyright (c) 2015, KU Leuven-ESAT-STADIUS, License & help @
    # http://www.esat.kuleuven.be/stadius/ADB/jumutc/softwareSALSA.php

    N = size(X,1)
    d = size(X,2) + 1
    check = issparse(X)

    if ~check
        w = rand(d)
        w = w./(sqrt(λ)*vecnorm(w))
        sub_arr = (I) -> append_ones(sub(X,I,:),k)
    else
        total = length(X.nzval)
        w = sprand(d,1,total/(N*d))
        w = w./(sqrt(λ)*vecnorm(w))
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

        # do a gradient descent step
        η_t = 1/(λ*t)
        w = (1 - η_t*λ).*w
        w = w - (η_t/k).*dfunc(At,yt,w_prev)
        # project back to the set B: w \in convex set B
        w = min(1,1/(sqrt(λ)*vecnorm(w))).*w

        # check the stopping criterion w.r.t. Tolerance, check, online_pass
        if online_pass == 0 && ~check && vecnorm(w - w_prev) < tolerance
            break
        end
    end

    w[1:end-1,:], w[end,:]
end
