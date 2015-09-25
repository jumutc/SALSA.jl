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

function mapstd{T <: Number}(datain::Matrix{T})
	# apply mapstd
	means = mean(datain,1)
	stds  = std(datain,1)
	stds[stds.==0] = 1

	dataout = datain - ones(size(datain,1),1)*means
	dataout = dataout./(ones(size(datain,1),1)*stds)
	(dataout, means, stds)
end

function mapstd(datain::SparseMatrixCSC)
	(datain, 0, 0)
end

function mapstd(datain::SparseMatrixCSC,mean,std)
	datain
end

function mapstd{T <: Number}(datain::Matrix{T},mean,std)
	# apply mapstd
	dataout = datain - ones(size(datain,1),1)*mean
	dataout = dataout./(ones(size(datain,1),1)*std)
	dataout
end