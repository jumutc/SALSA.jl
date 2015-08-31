using SALSA, Base.Test

sinc(x) = sin(x)./x

X = linspace(0.1,20,100)''
Xtest = linspace(0.11,19.9,100)''

y = sinc(X)

srand(1234)
model = SALSAModel(NONLINEAR,PEGASOS(),LEAST_SQUARES,
				   validation_criterion=MSE(),normalized=false,
				   process_labels=false,subset_size=3.)
model = salsa(X,y,model,Xtest)

@test_approx_eq_eps mse(sinc(Xtest), model.output.Ytest) 0.01 0.05


rand(1234)
model = SALSAModel(NONLINEAR,PEGASOS(),LEAST_SQUARES,
				   validation_criterion=MSE(),process_labels=false)
model = salsa(X,y,model,[])
Ytest = map_predict(model,Xtest)

@test_approx_eq_eps mse(sinc(Xtest), Ytest) 0.01 0.05