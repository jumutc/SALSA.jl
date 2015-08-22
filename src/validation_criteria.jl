validation_criteria(model::SALSAModel,X,Y,val_idx) = validation_criteria(model,sub(X,val_idx,:)[:,:],Y[val_idx])
validation_criteria{L <: Loss, A <: SGD}(model::SALSAModel{L,A},X,Y) = validation_criteria(model.validation_criteria,model,X,Y)
validation_criteria{L <: Loss, A <: RK_MEANS}(model::SALSAModel{L,A},X,Y) = validation_criteria(model.validation_criteria,model,X,Y)
validation_criteria{L <: Loss, A <: RDA}(model::SALSAModel{L,A},X,Y) = model.sparsity_cv*mean(model.output.w .!= 0) + (1-model.sparsity_cv)*validation_criteria(model.validation_criteria,model,X,Y)

validation_criteria{L <: Loss, A <: SGD}(criteria::MSE, model::SALSAModel{L,A}) = "mean squared error"
validation_criteria{L <: Loss, A <: SGD}(criteria::AUC, model::SALSAModel{L,A}) = "1 - AUC (area under curve)"
validation_criteria{L <: Loss, A <: SGD}(criteria::MISCLASS, model::SALSAModel{L,A}) = "misclassification rate"
validation_criteria{L <: Loss, A <: RK_MEANS}(criteria::SILHOUETTE, model::SALSAModel{L,A}) = "1 - mean(silhouette)"
validation_criteria{L <: Loss, A <: RDA, C <: Criteria}(criteria::C, model::SALSAModel{L,A}) = "weighted combination of: error/sparisty"

validation_criteria(criteria::MISCLASS,model,X,Y) 	= misclass(Y, predict_raw(model,X))
validation_criteria(criteria::MSE,model,X,Y) 	  	= mse(Y, predict_latent_raw(model,X))
validation_criteria(criteria::AUC,model,X,Y) 	  	= 1 - auc(Y, predict_latent_raw(model,X), n=criteria.n_thresholds)
validation_criteria(criteria::SILHOUETTE,model,X,Y) = begin
	assignments = predict_by_distance(model,X)
	dists = pairwise(model.algorithm.metric, X')
	cnts = counts(assignments,1:maximum(assignments))
	return 1 - mean(silhouettes(assignments,cnts,dists))
end
# validation_criteria(criteria::MISCLASS,model,X,Y) 	= begin
# 	if size(Y,2) == 1
# 		misclass(Y, predict_raw(model,X))
# 	else
# 		Y_ = membership(predict_latent_raw(model,X))
# 		misclass(Y_, membership(Y))
# 	end
# end