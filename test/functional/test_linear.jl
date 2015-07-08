using MAT, SALSA, Base.Test, MLBase

ripley = matread(joinpath(Pkg.dir("SALSA"),"data","ripley.mat"))

srand(1234)
model = salsa(LINEAR,PEGASOS,HINGE,ripley["X"],ripley["Y"],ripley["Xt"])
@test_approx_eq_eps mean(ripley["Yt"] .== model.output.Ytest) 0.89 0.01

srand(1234)
model = salsa(LINEAR,SIMPLE_SGD,HINGE,ripley["X"],ripley["Y"],ripley["Xt"])
@test_approx_eq_eps mean(ripley["Yt"] .== model.output.Ytest) 0.89 0.01

srand(1234)
model = SALSAModel(LINEAR,PEGASOS(),PINBALL,global_opt=DS([0,-1]))
model = salsa(ripley["X"],ripley["Y"],model,ripley["Xt"])
@test_approx_eq_eps mean(ripley["Yt"] .== model.output.Ytest) 0.89 0.01

srand(1234)
model = SALSAModel(LINEAR,PEGASOS(),LEAST_SQUARES,global_opt=DS([-1]))
model = salsa(ripley["X"],ripley["Y"],model,ripley["Xt"])
@test_approx_eq_eps mean(ripley["Yt"] .== model.output.Ytest) 0.885 0.01

srand(1234)
model = SALSAModel(LINEAR,PEGASOS(),LOGISTIC,global_opt=DS([-1]))
model = salsa(ripley["X"],ripley["Y"],model,ripley["Xt"])
@test_approx_eq_eps mean(ripley["Yt"] .== model.output.Ytest) 0.885 0.01

srand(1234)
model = SALSAModel(LINEAR,DROP_OUT(),HINGE,global_opt=DS([-10]))
model = salsa(ripley["X"],ripley["Y"],model,ripley["Xt"])
@test_approx_eq_eps mean(ripley["Yt"] .== model.output.Ytest) 0.88 0.02

srand(1234)
model = SALSAModel(LINEAR,L1RDA(),HINGE,global_opt=DS([-5,0,0]))
model = salsa(ripley["X"],ripley["Y"],model,ripley["Xt"])
@test_approx_eq_eps mean(ripley["Yt"] .== model.output.Ytest) 0.89 0.01

srand(1234)
model = SALSAModel(LINEAR,R_L1RDA(),HINGE,global_opt=DS([-5,0,0,-2]))
model = salsa(ripley["X"],ripley["Y"],model,ripley["Xt"])
@test_approx_eq_eps mean(ripley["Yt"] .== model.output.Ytest) 0.885 0.01

srand(1234)
model = SALSAModel(LINEAR,R_L2RDA(),HINGE,global_opt=DS([-5,0,0]))
model = salsa(ripley["X"],ripley["Y"],model,ripley["Xt"])
@test_approx_eq_eps mean(ripley["Yt"] .== model.output.Ytest) 0.88 0.01

srand(1234)
model = SALSAModel(LINEAR,ADA_L1RDA(),HINGE,global_opt=DS([-5,0,0]))
model = salsa(ripley["X"],ripley["Y"],model,ripley["Xt"])
@test_approx_eq_eps mean(ripley["Yt"] .== model.output.Ytest) 0.89 0.01