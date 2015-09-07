# Predict by evaluating a simple linear model
predict_raw(model::SALSAModel,X) = sign(predict_latent_raw(model,X))
predict_latent_raw(model::SALSAModel,X) = X*model.output.w .+ ones(size(X,1),1)*model.output.b
# aliases to predict according to validation criterion and task: regression/classification
predict(criterion::AUC, model::SALSAModel, X) 	  	= predict_raw(model, X)
predict(criterion::MISCLASS, model::SALSAModel, X) 	= predict_raw(model, X)
predict(criterion::MSE, model::SALSAModel, X) 	  	= predict_latent_raw(model, X)
predict(criterion::SILHOUETTE, model::SALSAModel, X) = predict_by_distance(model, X)

predict_raw(model::SALSAModel,f::DelimitedFile) = vcat([predict_raw(model,sub(f,i,:)) for i=1:count(f)]...)
predict_latent_raw(model::SALSAModel,f::DelimitedFile) = vcat([predict_latent_raw(model,sub(f,i,:)) for i=1:count(f)]...)

function predict(model::SALSAModel,X)
	if model.mode == LINEAR
  		predict(model.validation_criterion,model,X)
  	else
  		k = kernel_from_parameters(model.kernel,model.output.mode.k_params)
  		predict(model.validation_criterion,model,AFEm(model.output.mode.X_subset,k,X))
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
	mappings = convert(Array{Int},zeros(length(y)))
	mappings[x] = y
	mappings
end

# Map data to existing mean/std in the model and predict
function map_predict(model::SALSAModel,X) 
	if isdefined(model.output,:X_mean) && isdefined(model.output,:X_std) 
		X = mapstd(X,model.output.X_mean,model.output.X_std)
		predict(model,X)
	else
		predict(model,X)
	end
end
