using SALSA, Clustering, Distances, Base.Test

Xf = readcsv(joinpath(Pkg.dir("SALSA"),"data","iris.data.csv"))
Y = round(Int64, Xf[:,end])
X = Xf[:,1:end-1]

srand(1234)
w = stochastic_rk_means(X,RK_MEANS(PEGASOS,3,20),[1e-1],1,1000,1e-5)
dists = pairwise(Euclidean(), X', w)
(x,y) = findn(dists .== minimum(dists,2))
mappings = round(Int64, zeros(length(y)))
mappings[x] = y

@test_approx_eq_eps varinfo(length(unique(mappings)), mappings, 3, Y) .5 0.05