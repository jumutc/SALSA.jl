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

# Calculate the misclassification rate
misclass(y, yhat) = 1-mean(y.==yhat)
# Calculate the sum of squared differences between two vectors
sse(y, yhat) = norm(y-yhat)^2
# Calculates the average squared difference between the corresponding elements of two vectors
mse(y, yhat) = sse(y, yhat)/length(yhat)
# Area Under ROC surve with latent output y
auc(y, ylat; n=100) = std(ylat[:]) == 0 ? 0.0 : auc(roc(round(Int,y)[:], ylat[:], n))
# provide convenient function for parallalizing cross-validation
nfolds() = if nworkers() == 1 || nworkers() > 10 10 else nworkers() end
# helper function for AUC calculus
function auc(roc::Array{ROCNums{Int}})
	total_auc = zero(Float64)
	for i in length(roc):-1:2
		dy = true_positive_rate(roc[i-1]) - true_positive_rate(roc[i])
		dx = false_positive_rate(roc[i-1]) - false_positive_rate(roc[i])
		total_auc += ( dx*true_positive_rate(roc[i]) + 0.5*dx*dy )
	end
	total_auc
end