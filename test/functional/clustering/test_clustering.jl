using SALSA, Clustering, Distances, MLBase, Base.Test, Compat

Xf = readcsv(joinpath(Pkg.dir("SALSA"),"data","iris.data.csv"))
Y = round(Int64, Xf[:,end])
X = Xf[:,1:end-1]

srand(1234)
dummy = ones(length(Y),1)
model = SALSAModel(LINEAR,RK_MEANS(PEGASOS,3,20,Euclidean()),LEAST_SQUARES,
					validation_criteria=SILHOUETTE(),global_opt=DS([1]),
					cv_gen = @compat Nullable{CrossValGenerator}(Kfold(length(Y),3)))
model = salsa(X,dummy,model,X)
mappings = model.output.Ytest

@test_approx_eq_eps varinfo(length(unique(mappings)), mappings, 3, Y) .7 0.05