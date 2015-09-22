function ds(obj_func,pn)
    #
    # Internal function based on DFO-based randomized Directional Search
    #
    # Copyright (c) 2015 KU Leuven-ESAT-STADIUS, 
    #
    alpha = 1
    ft = obj_func(pn)
    pdim = length(pn)
    D = [eye(pdim) -eye(pdim)]
    x = pn

    @showprogress 1 "Running hyperparameter tuning... " for k=1:50
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