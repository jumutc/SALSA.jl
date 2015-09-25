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

AFEm(Xs, k::Kernel, X::DelimitedFile) = AFEm(Xs,k,sub(X,:,:))

function AFEm(Xs, k::Kernel, X)
    #
    # Automatic Feature Extraction by Nystrom method
    #
    # 
    (eigvals, eigvec) = eig_AFEm(Xs, k)
    AFEm(eigvals, eigvec, Xs, k, X)
end

function AFEm(eigvals, eigvec, Xs, k::Kernel, X)
    #
    # Automatic Feature Extraction by Nystrom method
    #
    features = kernel_matrix(k, X, Xs) * eigvec
    features .* repmat(1./sqrt(eigvals'),size(X,1),1)
end

function eig_AFEm(Xs, k::Kernel)
    # eigenvalue decomposition to do...
    omega = kernel_matrix(k, Xs)
    (eigvals, eigvec) = eig(omega + 2.*eye(size(Xs,1)))
    indices = find(x -> x > eps(), eigvals)
    eigvals[indices], eigvec[:,indices]
end