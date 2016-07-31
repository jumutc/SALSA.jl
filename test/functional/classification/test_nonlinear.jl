using MAT, SALSA, Base.Test, MLBase

ripley = matread(joinpath(dirname(@__FILE__),"..","..","..","data","ripley.mat"))

srand(1234)
model = salsa(NONLINEAR,PEGASOS,HINGE,ripley["X"],ripley["Y"],ripley["Xt"])
@test_approx_eq_eps mean(ripley["Yt"] .== model.output.Ytest) 0.89 0.1

srand(1234)
model = SALSAModel(NONLINEAR,PEGASOS(),LOGISTIC,
	global_opt=DS([-1,-1,1]),kernel=PolynomialKernel,online_pass=10)
model = salsa(ripley["X"],ripley["Y"],model,ripley["Xt"])
@test_approx_eq_eps mean(ripley["Yt"] .== model.output.Ytest) 0.89 0.1

srand(1234)
model = SALSAModel(NONLINEAR,PEGASOS(),LOGISTIC,
	global_opt=DS([-1]),kernel=LinearKernel,online_pass=10)
model = salsa(ripley["X"],ripley["Y"],model,ripley["Xt"])
@test_approx_eq_eps mean(ripley["Yt"] .== model.output.Ytest) 0.89 0.1

srand(1234)
model = SALSAModel(NONLINEAR,L1RDA(),HINGE,global_opt=DS([-5,0,0,1]),online_pass=10)
model = salsa(ripley["X"],ripley["Y"],model,ripley["Xt"])
@test_approx_eq_eps mean(ripley["Yt"] .== model.output.Ytest) 0.89 0.1

srand(1234)
model = SALSAModel(NONLINEAR,ADA_L1RDA(),HINGE,global_opt=DS([-5,0,0,1]),online_pass=10)
model = salsa(ripley["X"],ripley["Y"],model,ripley["Xt"])
@test_approx_eq_eps mean(ripley["Yt"] .== model.output.Ytest) 0.89 0.1