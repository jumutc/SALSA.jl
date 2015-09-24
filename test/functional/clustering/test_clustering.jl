using SALSA, Clustering, Distances, MLBase, Base.Test, Compat

Xf = readcsv(joinpath(Pkg.dir("SALSA"),"data","iris.data.csv"))
Y = convert(Array{Int}, Xf[:,end])
dY = @compat Array{Int}(length(Y))
X = Xf[:,1:end-1]

srand(1234)
model = SALSAModel(LINEAR,RK_MEANS(PEGASOS,3,20,Euclidean()),LEAST_SQUARES,
					validation_criterion=SILHOUETTE(),global_opt=DS([-1]),process_labels=false,
					cv_gen = @compat Nullable{CrossValGenerator}(Kfold(length(Y),3)))
model = salsa(X,dY,model,X)
mappings = model.output.Ytest

@test_approx_eq_eps varinfo(length(unique(mappings)), mappings, 3, Y) .7 0.3

srand(1234)
model = SALSAModel(LINEAR,RK_MEANS(ADA_L1RDA,3,20,CosineDist()),HINGE,
					validation_criterion=SILHOUETTE(),global_opt=DS([-5,-5,-1]),process_labels=false,
					cv_gen = @compat Nullable{CrossValGenerator}(Kfold(length(Y),3)))
model = salsa(sparse(X),dY,model,X)
mappings = model.output.Ytest

if VERSION >= v"0.4"
	@test_approx_eq_eps varinfo(length(unique(mappings)), mappings, 3, Y) .5 0.3
end