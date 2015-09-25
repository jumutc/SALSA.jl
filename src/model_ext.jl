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

# extensive set of multiplicated aliases for different algorithms and models
model_from_parameters{M <: Mode, A <: SGD, L <: NonParametricLoss}(model::SALSAModel{L,A,M},params) = begin 
	model.output.dfunc = loss_derivative(model.loss_function) 
    model.output.alg_params = [exp(params[1])]
    model 
end
model_from_parameters{M <: Mode, A <: SGD}(model::SALSAModel{PINBALL,A,M},params) = begin 
    model.output.dfunc = loss_derivative(model.loss_function,exp(params[1])) 
    model.output.alg_params = [exp(params[2])]
    model 
end
model_from_parameters{M <: Mode, A <: RDA, L <: NonParametricLoss}(model::SALSAModel{L,A,M},params) = begin 
	model.output.dfunc = loss_derivative(model.loss_function) 
    model.output.alg_params = exp(params[1:3])
    model 
end
model_from_parameters{M <: Mode, A <: RDA}(model::SALSAModel{PINBALL,A,M},params) = begin 
    model.output.dfunc = loss_derivative(model.loss_function,exp(params[1])) 
    model.output.alg_params = exp(params[2:4])
    model 
end
model_from_parameters{M <: Mode, L <: NonParametricLoss}(model::SALSAModel{L,R_L1RDA,M},params) = begin 
    model.output.dfunc = loss_derivative(model.loss_function) 
    model.output.alg_params = exp(params[1:4])
    model 
end
model_from_parameters{M <: Mode}(model::SALSAModel{PINBALL,R_L1RDA,M},params) = begin 
    model.output.dfunc = loss_derivative(model.loss_function,exp(params[1])) 
    model.output.alg_params = exp(params[2:5])
    model 
end
model_from_parameters{M <: Mode, A <: RK_MEANS, L <: Loss}(model::SALSAModel{L,A,M},params) = begin 
    if model.algorithm.support_alg <: SGD
        model.output.alg_params = [exp(params[1])]
    elseif model.algorithm.support_alg == R_L1RDA
        model.output.alg_params = exp(params[1:4])
    else
        model.output.alg_params = exp(params[1:3])
    end
    model 
end