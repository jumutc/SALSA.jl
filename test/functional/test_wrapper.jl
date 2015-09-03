using MAT, SALSA, Base.Test, MLBase

X = DelimitedFile(joinpath(Pkg.dir("SALSA"),"data","iris.data.csv"),false,',')
Xf = readcsv(joinpath(Pkg.dir("SALSA"),"data","iris.data.csv"))
Y = Xf[:,end]; Y[Y.>1] = -1

srand(1234)
model = salsa(X, Y, SALSAModel(LINEAR,SIMPLE_SGD(),HINGE,global_opt=DS([-1])), X)
@test_approx_eq mean(Y .== model.output.Ytest) 1.0

srand(1234)
model = salsa(X, Y, SALSAModel(LINEAR,SIMPLE_SGD(),HINGE,global_opt=DS([-1]),online_pass=10), Xf)
@test_approx_eq_eps mean(Y .== model.output.Ytest) 0.9 0.1

srand(1234)
model = salsa(NONLINEAR, PEGASOS, HINGE, X, Y, X)
@test_approx_eq mean(Y .== model.output.Ytest) 1.0