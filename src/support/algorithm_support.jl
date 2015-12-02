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

function fix_space(train_idx, N::Int)
    if ~isempty(train_idx)
        (train_idx, size(train_idx,1))
    else
        (collect(1:1:N), N)
    end
end

function fix_sampling(online_pass, N::Int)
    if online_pass > 0
        (t,k) -> begin
            s = t % N
            s > 0 ? s : N
        end
    else
        pd = Categorical(N)
        (t,k) -> rand(pd,k)
    end
end

function fix_iter(online_pass, N::Int, max_iter::Int)
    online_pass > 0 ? N*online_pass : max_iter
end
