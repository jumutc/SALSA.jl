using MAT, SALSA, Base.Test, MLBase

X = DelimitedFile(joinpath(dirname(@__FILE__),"..","..","data","iris.data.csv"),false,',')
Xf = readcsv(joinpath(dirname(@__FILE__),"..","..","data","iris.data.csv"))
Y = Xf[:,end]; Y[Y.>1] = -1

w,b = pegasos_alg(loss_derivative(HINGE),X,Y,1.,1,1,1e-5,10)
@test_approx_eq_eps mean(sign(Xf*w .+ b) .== Y) 0.9 0.1