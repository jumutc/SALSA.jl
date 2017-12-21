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

abstract type QANode end

type LinearQANode <: QANode
	question:: String
	procfunc:: Function
	# field not provided by default at instantiation
	options:: Dict
	answer:: String
end

immutable QAOption{N <: QANode}
	updatefunc:: Function
	next:: Nullable{N}
end

LinearQANode(question, procfunc) = LinearQANode(question,procfunc,Dict(),"") 