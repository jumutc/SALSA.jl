# Predict by evaluating a simple linear model
predict_raw(model::SALSAModel,X) = sign(predict_latent_raw(model,X))
predict_latent_raw(model::SALSAModel,X) = X*model.output.w .+ ones(size(X,1),1)*model.output.b
# aliases to predict according to validation criteria and task: regression/classification
predict(criteria::AUC, model::SALSAModel, X) 	  	= predict(model, X)
predict(criteria::MISCLASS, model::SALSAModel, X) 	= predict(model, X)
predict(criteria::MSE, model::SALSAModel, X) 	  	= predict_latent(model, X)
predict(criteria::SILHOUETTE, model::SALSAModel, X) = predict_by_distance(model, X)

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

function predict_by_distance(model::SALSAModel,X)
	if model.mode == NONLINEAR
		k = kernel_from_parameters(model.kernel,model.output.mode.k_params)
		X = AFEm(model.output.mode.X_subset,k,X)
	end

	dists = pairwise(model.algorithm.metric, X', model.output.w)
	(x,y) = findn(dists .== minimum(dists,2))
	mappings = round(Int64, zeros(length(y)))
	mappings[x] = y
	mappings
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