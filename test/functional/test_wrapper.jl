using MAT, SALSA, Base.Test, MLBase

X = DelimitedFile(joinpath(Pkg.dir("SALSA"),"data","iris.data.csv"),false,',')
Xf = readcsv(joinpath(Pkg.dir("SALSA"),"data","iris.data.csv"))
Y = Xf[:,end]; Y[Y.>1] = -1

srand(1234)
model = salsa(X, Y, SALSAModel(LINEAR,SIMPLE_SGD(),HINGE,global_opt=DS([-1])), Xf)
@test_approx_eq mean(Y .== model.output.Ytest) 1.0