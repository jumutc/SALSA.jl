using MAT, SALSA, Base.Test

data = matread(joinpath("..","data","ripley.mat"))
model = salsa(data["X"],data["Y"],data["Xt"])

@test_approx_eq_eps 1-mean(data["Yt"] .== model.Ytest) 0.12 2e-2