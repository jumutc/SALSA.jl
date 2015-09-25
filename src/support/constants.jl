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

abstract Criterion 
abstract CCriterion <: Criterion
immutable MSE <: CCriterion end
immutable MISCLASS <: CCriterion end 
immutable SILHOUETTE <: Criterion end 
immutable AUC <: CCriterion
	n_thresholds::Integer
end

abstract GlobalOpt 
immutable CSA <: GlobalOpt end 
immutable DS <: GlobalOpt
	init_params::Vector 
end

DS() = DS(Array(Float64,0))
AUC() = AUC(100)