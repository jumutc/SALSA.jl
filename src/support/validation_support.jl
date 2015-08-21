# Calculate the misclassification rate
misclass(y, yhat) = 1-mean(y.==yhat)
# Calculate the sum of squared differences between two vectors
sse(y, yhat) = norm(y-yhat)^2
# Calculates the average squared difference between the corresponding elements of two vectors
mse(y, yhat) = sse(y, yhat)/length(yhat)
# Area Under ROC surve with latent output y
auc(y, ylat; n=100) = std(ylat[:]) == 0 ? 0.0 : auc(roc(int(y)[:], ylat[:], n))
# provide convenient function for parallalizing cross-validation
nfolds() = if nworkers() == 1 || nworkers() > 10 10 else nworkers() end
# helper function for AUC calculus
function auc(roc::Array{ROCNums{Int}})
	total_auc = zero(Float64)
	for i in length(roc):-1:2
		dy = true_positive_rate(roc[i-1]) - true_positive_rate(roc[i])
		dx = false_positive_rate(roc[i-1]) - false_positive_rate(roc[i])
		total_auc += ( dx*true_positive_rate(roc[i]) + 0.5*dx*dy )
	end
	total_auc
end