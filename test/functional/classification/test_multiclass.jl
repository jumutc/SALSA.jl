using SALSA, Base.Test

Xf = readcsv(joinpath(dirname(@__FILE__),"..","..","..","data","iris.data.csv"))
X = Xf[:,1:end-1]
Y = Xf[:,end]

srand(1234)
model = salsa(LINEAR,PEGASOS,HINGE,X,Y,X)
@test_approx_eq_eps mean(Y .== model.output.Ytest) 0.85 0.2