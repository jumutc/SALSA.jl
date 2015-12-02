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

function entropy_subset{K <: Kernel}(X, k::K, subset_size::Float64)
    # Performs Renyi Entropy based representative points selection
    capacity = round(Integer,subset_size)
    selectionset = randperm(size(X,1))
    sv = selectionset[1:capacity]
    svX = sub(X,sv,:)

    # Calculating the Renyi for pre-determined points
    totalinfo2 = sum(kernel_matrix(k,svX),2)
    totalinfo1 = 1:1:capacity

    capsquare = capacity^2
    totalcrit = sum(totalinfo2)
    logtotalcrit = -log(totalcrit/capsquare)

    max_c = logtotalcrit
    # Maximizing the quadratic Renyi Entropy
    for i=1:size(X,1)
        # Find the smallest entropy
        val, id = findmin(totalinfo2)
        # Subtract from totalcrit once for row and once for column
        # and add 1 for diagonal term which is subtracted twice
        temptotalcrit = totalcrit - 2*val + 1
        #Try to evaluate kernel function
        if (id == capacity)
            subi = totalinfo1[1:id-1]
        elseif (id == 1)
            subi = totalinfo1[id+1:end]
        else
            subi = totalinfo1[[1:id-1;id+1:end]]
        end
        distance_eval = sum(kernel_matrix(k,sub(X,i,:),svX[subi,:]))
        # Add to totalcrit once for row and once for column
        # and subtract 1 for diagonal term which is added twice
        temptotalcrit = temptotalcrit + 2*distance_eval - 1
        logtemptotalcrit = -log(temptotalcrit/capsquare)
        # Evaluate that Renyi Entropy has increased
        if (max_c <= logtemptotalcrit)
            totalinfo2[id] = distance_eval
            totalcrit = temptotalcrit
            max_c = logtemptotalcrit
            sv[id] = i
        end
    end

    sv
end
