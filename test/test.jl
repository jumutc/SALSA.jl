using MAT

vars = matread(joinpath("..","data","ripley.mat"))
X = vars["X"]
Y = vars["Y"]
Y[Y .> 1] = -1

using SALSA
@time model = salsa(X,Y,X)

@printf "\nTraining error=%.5f" 1-mean(Y .== model.Ytest)
@printf "\nSparsity=%.5f" mean(model.w .== 0)