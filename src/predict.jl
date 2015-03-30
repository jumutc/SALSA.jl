export predict, map_predict
# Predict by evaluating a simple linear model
predict(model::SALSAModel,X) = sign(X* model.w .+ model.b)
# Map data to existing mean/std in the model and predict
function map_predict(model::SALSAModel,X) 
	if ~isempty(model.X_mean) && ~isempty(model.X_mean) 
		X = mapstd(X,model.X_mean,model.X_std)
		predict(model,X)
	else
		predict(model,X)
	end
end