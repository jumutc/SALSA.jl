# extensive set of multiplicated aliases for different algorithms and models
model_from_parameters{M <: Mode, A <: SGD, L <: NonParametricLoss}(model::SALSAModel{L,A,M},params) = begin 
	model.dfunc = loss_derivative(model.loss_function) 
    model.alg_params = [exp(params[1])]
    model 
end
model_from_parameters{M <: Mode, A <: SGD}(model::SALSAModel{PINBALL,A,M},params) = begin 
    model.dfunc = loss_derivative(model.loss_function,exp(params[1])) 
    model.alg_params = [exp(params[2])]
    model 
end
model_from_parameters{M <: Mode, A <: RDA, L <: NonParametricLoss}(model::SALSAModel{L,A,M},params) = begin 
	model.dfunc = loss_derivative(model.loss_function) 
    model.alg_params = exp(params[1:3])
    model 
end
model_from_parameters{M <: Mode, A <: RDA}(model::SALSAModel{PINBALL,A,M},params) = begin 
    model.dfunc = loss_derivative(model.loss_function,exp(params[1])) 
    model.alg_params = [exp(params[2:4])]
    model 
end
model_from_parameters{M <: Mode, L <: NonParametricLoss}(model::SALSAModel{L,R_L1RDA,M},params) = begin 
    model.dfunc = loss_derivative(model.loss_function) 
    model.alg_params = exp(params[1:4])
    model 
end
model_from_parameters{M <: Mode}(model::SALSAModel{PINBALL,R_L1RDA,M},params) = begin 
    model.dfunc = loss_derivative(model.loss_function,exp(params[1])) 
    model.alg_params = [exp(params[2:5])]
    model 
end


ds_parameters_from_model{L <: NonParametricLoss}(model::SALSAModel{L,PEGASOS,LINEAR}) = [-5]
ds_parameters_from_model(model::SALSAModel{PINBALL,PEGASOS,LINEAR}) = [0,-5]
ds_parameters_from_model{L <: NonParametricLoss}(model::SALSAModel{L,DROP_OUT,LINEAR}) = [1]
ds_parameters_from_model(model::SALSAModel{PINBALL,DROP_OUT,LINEAR}) = [0,1]
ds_parameters_from_model{A <: RDA, L <: NonParametricLoss}(model::SALSAModel{L,A,LINEAR}) = [-5;ones(2)]
ds_parameters_from_model{A <: RDA}(model::SALSAModel{PINBALL,A,LINEAR}) = [0;-5;ones(2)]
ds_parameters_from_model{L <: NonParametricLoss}(model::SALSAModel{L,R_L1RDA,LINEAR}) = [-5;ones(3)]
ds_parameters_from_model(model::SALSAModel{PINBALL,R_L1RDA,LINEAR}) = [0;-5;ones(3)]
ds_parameters_from_model{L <: NonParametricLoss}(model::SALSAModel{L,R_L2RDA,LINEAR}) = [-5,1,-2]
ds_parameters_from_model(model::SALSAModel{PINBALL,R_L2RDA,LINEAR}) = [0,-5,1,-2]

ds_parameters_from_model{L <: NonParametricLoss}(model::SALSAModel{L,PEGASOS,NONLINEAR}) = [-5;randn(1)]
ds_parameters_from_model(model::SALSAModel{PINBALL,PEGASOS,NONLINEAR}) = [0;-5;randn(1)]
ds_parameters_from_model{L <: NonParametricLoss}(model::SALSAModel{L,DROP_OUT,NONLINEAR}) = [1;randn(1)]
ds_parameters_from_model(model::SALSAModel{PINBALL,DROP_OUT,NONLINEAR}) = [0;1;randn(1)]
ds_parameters_from_model{A <: RDA, L <: NonParametricLoss}(model::SALSAModel{L,A,NONLINEAR}) = [-5;ones(2);randn(1)]
ds_parameters_from_model{A <: RDA}(model::SALSAModel{PINBALL,A,NONLINEAR}) = [0;-5;ones(2);randn(1)]
ds_parameters_from_model{L <: NonParametricLoss}(model::SALSAModel{L,R_L1RDA,NONLINEAR}) = [-5;ones(3);randn(1)]
ds_parameters_from_model(model::SALSAModel{PINBALL,R_L1RDA,NONLINEAR}) = [0;-5;ones(3);randn(1)]
ds_parameters_from_model{L <: NonParametricLoss}(model::SALSAModel{L,R_L2RDA,NONLINEAR}) = [-5;1;-2;randn(1)]
ds_parameters_from_model(model::SALSAModel{PINBALL,R_L2RDA,NONLINEAR}) = [0;-5;1;-2;randn(1)]