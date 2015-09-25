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

function ds(obj_func,pn)
    #
    # Internal function based on DFO-based randomized Directional Search
    #
    alpha = 1
    ft = obj_func(pn)
    pdim = length(pn)
    D = [eye(pdim) -eye(pdim)]
    x = pn

    @showprogress 1 "Running hyperparameter tuning... " 20 for k=1:50
       # restart polling directions
       dk = randperm(pdim*2)
       ft_old = ft
        
       result = @parallel (vcat) for i=1:2*pdim
          f_new = obj_func(x + alpha .* D[:, dk[i]])
          x_new = x + alpha .* D[:, dk[i]]
          [x_new' f_new]
       end

       ind_min = indmin(result[:,end])
       if (result[ind_min,end] < ft - 1e-5 * alpha^2)
          x = vec(result[ind_min,1:end-1])
          ft = result[ind_min,end]
       end
       
       # exit on ɛ-based criterion & number of try outs 
       if ft - ft_old < 1e-5
           break
       end
       
       # decrease step size if needed
       if (mean(result[:,end]) >= ft_old - 1e-5 * alpha^2)
          alpha = 0.5 * alpha
       end
    end

    ft, x
end