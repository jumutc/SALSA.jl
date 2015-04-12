export predict, map_predict

# Predict by evaluating a simple linear model
predict_raw(model::SALSAModel,X) = sign(X*model.output.w .+ model.output.b)

function predict(model::SALSAModel,X)
	if model.mode == LINEAR
  		predict_raw(model,X)
  	else
  		k = kernel_from_parameters(model.kernel,model.output.mode.k_params)
  		predict_raw(model,AFEm(model.output.mode.X_subset,k,X))
  	end	
end

# Map data to existing mean/std in the model and predict
function map_predict(model::SALSAModel,X) 
	if ~isempty(model.X_mean) && ~isempty(model.X_mean) 
		X = mapstd(X,model.X_mean,model.X_std)
		predict(model,X)
	else
		predict(model,X)
	end
end