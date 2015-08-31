validation_criterion(model::SALSAModel,X,Y,val_idx) = validation_criterion(model,sub(X,val_idx,:)[:,:],Y[val_idx])
validation_criterion{L <: Loss, A <: SGD}(model::SALSAModel{L,A},X,Y) = validation_criterion(model.validation_criterion,model,X,Y)
validation_criterion{L <: Loss, A <: RK_MEANS}(model::SALSAModel{L,A},X,Y) = validation_criterion(model.validation_criterion,model,X,Y)
validation_criterion{L <: Loss, A <: RDA}(model::SALSAModel{L,A},X,Y) = model.sparsity_cv*mean(model.output.w .!= 0) + (1-model.sparsity_cv)*validation_criterion(model.validation_criterion,model,X,Y)

validation_criterion{L <: Loss, A <: SGD}(criterion::MSE, model::SALSAModel{L,A}, val) = ["mean squared error"; val]
validation_criterion{L <: Loss, A <: SGD}(criterion::AUC, model::SALSAModel{L,A}, val) = ["AUC (area under curve)"; 1 - val]
validation_criterion{L <: Loss, A <: SGD}(criterion::MISCLASS, model::SALSAModel{L,A}, val) = ["misclassification rate"; val]
validation_criterion{L <: Loss, A <: RK_MEANS}(criterion::SILHOUETTE, model::SALSAModel{L,A}, val) = ["mean silhouette"; 1 - val]
validation_criterion{L <: Loss, A <: RDA, C <: Criterion}(criterion::C, model::SALSAModel{L,A}, val) = ["weighted combination of: error/sparsity"; val]

validation_criterion(criterion::MISCLASS,model,X,Y) 	= misclass(Y, predict_raw(model,X))
validation_criterion(criterion::MSE,model,X,Y) 	  	= mse(Y, predict_latent_raw(model,X))
validation_criterion(criterion::AUC,model,X,Y) 	  	= 1 - auc(Y, predict_latent_raw(model,X), n=criterion.n_thresholds)
validation_criterion(criterion::SILHOUETTE,model,X,Y) = begin
	asgs = convert(Array{Int},predict_by_distance(model,X))
	cnts = convert(Array{Int},counts(asgs,1:maximum(asgs)))
	dists = pairwise(model.algorithm.metric, X')
	return 1 - mean(silhouettes(asgs,cnts,dists))
end
# validation_criterion(criterion::MISCLASS,model,X,Y) 	= begin
# 	if size(Y,2) == 1
# 		misclass(Y, predict_raw(model,X))
# 	else
# 		Y_ = membership(predict_latent_raw(model,X))
# 		misclass(Y_, membership(Y))
# 	end
# end