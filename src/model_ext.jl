# extensive set of multiplicated aliases for different algorithms and models
model_from_parameters{M <: Mode, A <: SGD, L <: NonParametricLoss}(model::SALSAModel{L,A,M},params) = begin 
	model.output.dfunc = loss_derivative(model.loss_function) 
    model.output.alg_params = [exp(params[1])]
    model 
end
model_from_parameters{M <: Mode, A <: SGD, L <: NonParametricLoss}(model::SALSAModel{L,RK_MEANS{A},M},params) = begin 
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
model_from_parameters{M <: Mode, A <: RDA, L <: NonParametricLoss}(model::SALSAModel{L,RK_MEANS{A},M},params) = begin 
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
    else
        model.output.alg_params = exp(params[1:3])
    end
    model 
end