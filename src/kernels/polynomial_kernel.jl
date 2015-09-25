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

immutable PolynomialKernel <: Kernel
    tau::Float64
    d::Float64
end

function kernel_matrix(k::PolynomialKernel, Xr::Matrix, Xc::Matrix)
    nXr = size(Xr,1)
    nXc = size(Xc,1)
    K = Xr*Xc'
    for j=1:nXc, i=1:nXr
        K[i,j] = (K[i,j] + k.tau)^ceil(k.d)
    end
    K
end