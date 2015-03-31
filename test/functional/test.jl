using MAT, SALSA, Base.Test

data = matread(joinpath(Pkg.dir("SALSA"),"data","ripley.mat"))

model = salsa(PEGASOS,LINEAR,HINGE,data["X"],data["Y"],data["Xt"])
@test_approx_eq_eps mean(data["Yt"] .== model.Ytest) 0.88 5e-2

model = salsa(PEGASOS,LINEAR,PINBALL,data["X"],data["Y"],data["Xt"])
@test_approx_eq_eps mean(data["Yt"] .== model.Ytest) 0.88 5e-2

model = salsa(PEGASOS,LINEAR,LOGISTIC,data["X"],data["Y"],data["Xt"])
@test_approx_eq_eps mean(data["Yt"] .== model.Ytest) 0.88 5e-2