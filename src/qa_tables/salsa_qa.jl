function salsa_qa(X)
	@printf "\nDo you have any target varaible of interest in X of type %s: " typeof(X) 
	ans = read(STDIN,Char)

	if ans == 'y' || ans == 'Y'

	else
		@printf "\nWe think you have a clustering problem"
		@printf "\nPlease specify the numer of clusters you want to extract: "
		num_c = convert(Int64, read(STDIN,Char))

		dummy = ones(length(Y),1)
		model = SALSAModel(LINEAR,RK_MEANS(PEGASOS,num_c,20,Euclidean()),LEAST_SQUARES,
							validation_criteria=SILHOUETTE(),global_opt=DS([-1]))
		model = salsa(X,dummy,model,X)

		return model.output.Ytest
	end
end