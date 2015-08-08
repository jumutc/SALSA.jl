function salsa_qa(X)
	@printf "\nDo you have any target varaible of interest in X of type %s: " typeof(X) 
	ans = readline(STDIN)

	if startswith(ans,"y") || startswith(ans,"Y")
		@printf "\nPlease provide the column number of your target variable: "
		col_num = parse(Int,readline(STDIN))

		if length(unique(X[:,col_num])) > 10
			cols = Set(1:size(X,2)); delete!(cols,col_num); cols = collect(cols)  
			@printf "\nWe think you have a regression problem (>10 unique values in target)"
			@printf "\nIs this assumption correct: "
			ans = readline(STDIN)
			if startswith(ans,"y") || startswith(ans,"Y")
				@printf "\nComputing the model... "
				model = SALSAModel(NONLINEAR,PEGASOS(),LEAST_SQUARES,
				   				   validation_criteria=MSE(),subset_size=3.)
				model = salsa(X[:,cols],X[:,col_num],model,Array{Float64}(0,0))
				return model, Array(Float64,0)
			else
				model = salsa(X[:,cols],X[:,col_num])
				return model, Array(Int64,0)
			end
		else
			@printf "\nWe think you have a classification problem, computing..."
			cols = Set(1:size(X,2)); delete!(cols,col_num); cols = collect(cols) 
			model = salsa(X[:,cols],X[:,col_num])
			return model, Array(Int64,0)
		end
	else
		@printf "\nWe think you have a clustering problem"
		@printf "\nPlease specify the numer of clusters you want to extract: "
		num_c = parse(Int,readline(STDIN))

		@printf "\nComputing the model... "
		dummy = ones(size(X,1),1)
		model = SALSAModel(LINEAR,RK_MEANS(PEGASOS,num_c,20,Euclidean()),LEAST_SQUARES,
						   validation_criteria=SILHOUETTE(),global_opt=DS([-1]))
		model = salsa(X,dummy,model,X)

		return model, model.output.Ytest
	end
end