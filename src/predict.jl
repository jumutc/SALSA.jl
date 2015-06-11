export predict, predict_latent, map_predict, map_predict_latent

# Predict by evaluating a simple linear model
predict_raw(model::SALSAModel,X) = sign(predict_latent(model,X))
predict_latent_raw(model::SALSAModel,X) = X*model.output.w .+ model.output.b

function predict(model::SALSAModel,X)
	if model.mode == LINEAR
  		predict_raw(model,X)
  	else
  		k = kernel_from_parameters(model.kernel,model.output.mode.k_params)
  		predict_raw(model,AFEm(model.output.mode.X_subset,k,X))
  	end	
end

function predict_latent(model::SALSAModel,X)
	if model.mode == LINEAR
  		predict_latent_raw(model,X)
  	else
  		k = kernel_from_parameters(model.kernel,model.output.mode.k_params)
  		predict_latent_raw(model,AFEm(model.output.mode.X_subset,k,X))
  	end	
end

# Map data to existing mean/std in the model and predict
function map_predict(model::SALSAModel,X) 
	if ~isempty(model.output.X_mean) && ~isempty(model.output.X_mean) 
		X = mapstd(X,model.output.X_mean,model.output.X_std)
		predict(model,X)
	else
		predict(model,X)
	end
end

# Map data to existing mean/std in the model and predict
function map_predict_latent(model::SALSAModel,X) 
	if ~isempty(model.output.X_mean) && ~isempty(model.output.X_mean) 
		X = mapstd(X,model.output.X_mean,model.output.X_std)
		predict_latent(model,X)
	else
		predict_latent(model,X)
	end
end