using MAT, SALSA, Base.Test, MLBase

ripley = matread(joinpath(Pkg.dir("SALSA"),"data","ripley.mat"))

(Xtrain,mean_,std_) = mapstd(ripley["X"])
Xtest = mapstd(ripley["Xt"],mean_,std_)

Xtrain = sparse(Xtrain)
Xtest = sparse(Xtest)

srand(1234)
model = salsa(LINEAR,PEGASOS,HINGE,Xtrain,ripley["Y"],Xtest)
@test_approx_eq_eps mean(ripley["Yt"] .== model.output.Ytest) 0.89 0.1

srand(1234)
model = salsa(LINEAR,SIMPLE_SGD,HINGE,Xtrain,ripley["Y"],Xtest)
@test_approx_eq_eps mean(ripley["Yt"] .== model.output.Ytest) 0.89 0.1

srand(1234)
model = SALSAModel(LINEAR,PEGASOS(),HINGE,global_opt=DS([-1]),validation_criterion=AUC(100))
model = salsa(Xtrain,ripley["Y"],model,[])
Ytest = map_predict(model,Xtest)
@test_approx_eq_eps mean(ripley["Yt"] .== Ytest) 0.8 0.1

srand(1234)
model = SALSAModel(LINEAR,PEGASOS(),LEAST_SQUARES,global_opt=DS([-1]))
model = salsa(Xtrain,ripley["Y"],model,Xtest)
@test_approx_eq_eps mean(ripley["Yt"] .== model.output.Ytest) 0.89 0.1

srand(1234)
model = SALSAModel(LINEAR,PEGASOS(),LOGISTIC,global_opt=DS([-5]))
model = salsa(Xtrain,ripley["Y"],model,Xtest)
@test_approx_eq_eps mean(ripley["Yt"] .== model.output.Ytest) 0.89 0.1

if VERSION < v"0.5-" # remove when sparse vectors/natrices are fixed
  srand(1234)
  model = SALSAModel(LINEAR,DROP_OUT(),HINGE,global_opt=DS([-5]))
  model = salsa(Xtrain,ripley["Y"],model,Xtest)
  @test_approx_eq_eps mean(ripley["Yt"] .== model.output.Ytest) 0.89 0.1
end

srand(1234)
model = SALSAModel(LINEAR,L1RDA(),HINGE,global_opt=DS([-5,0,0]))
model = salsa(Xtrain,ripley["Y"],model,Xtest)
@test_approx_eq_eps mean(ripley["Yt"] .== model.output.Ytest) 0.85 0.1

if VERSION < v"0.5-" # remove when sparse vectors/natrices are fixed 
  srand(1234)
  model = SALSAModel(LINEAR,R_L1RDA(),HINGE,global_opt=DS([-5,0,0,-2]))
  model = salsa(Xtrain,ripley["Y"],model,Xtest)
  @test_approx_eq_eps mean(ripley["Yt"] .== model.output.Ytest) 0.85 0.1

  srand(1234)
  model = SALSAModel(LINEAR,R_L2RDA(),HINGE,global_opt=DS([-5,0,0]))
  model = salsa(Xtrain,ripley["Y"],model,Xtest)
  @test_approx_eq_eps mean(ripley["Yt"] .== model.output.Ytest) 0.85 0.1

  srand(1234)
  model = SALSAModel(LINEAR,ADA_L1RDA(),HINGE,global_opt=DS([-5,0,0]))
  model = salsa(Xtrain,ripley["Y"],model,Xtest)
  @test_approx_eq_eps mean(ripley["Yt"] .== model.output.Ytest) 0.85 0.1
end
