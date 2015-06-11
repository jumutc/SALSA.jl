validation_criteria(model::SALSAModel,X,Y,val_idx) = validation_criteria(model,X[val_idx,:],Y[val_idx])
validation_criteria{L <: Loss, A <: SGD}(model::SALSAModel{L,A},X,Y) = validation_criteria(model.validation_criteria,model,X,Y)
validation_criteria{L <: Loss, A <: RDA}(model::SALSAModel{L,A},X,Y) = model.sparsity_cv*mean(model.output.w .!= 0) + (1-model.sparsity_cv)*validation_criteria(model.validation_criteria,model,X,Y)

validation_criteria{L <: Loss, A <: SGD}(model::SALSAModel{L,A}) = typeof(model.validation_criteria) == AUC ? "1-auc" : "misclassification rate"
validation_criteria{L <: Loss, A <: RDA}(model::SALSAModel{L,A}) = "weighted combination of: error/sparisty"

validation_criteria(criteria::MISCLASS,model,X,Y) = misclass(Y, predict_raw(model,X))
validation_criteria(criteria::AUC,model,X,Y) = 1 - auc(Y, predict_latent(model,X), n=criteria.n_thresholds)