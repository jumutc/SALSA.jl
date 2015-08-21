using SALSA, Base.Test

sinc(x) = sin(x)./x

X = linspace(0.1,20,100)''
Xtest = linspace(0.11,19.9,100)''

y = sinc(X)

srand(1234)
model = SALSAModel(NONLINEAR,PEGASOS(),LEAST_SQUARES,
				   validation_criteria=MSE(),normalized=false,
				   process_labels=false,subset_size=3.)
model = salsa(X,y,model,Xtest)

@test_approx_eq_eps mse(sinc(Xtest), model.output.Ytest) 0.01 0.01


rand(1234)
model = SALSAModel(NONLINEAR,PEGASOS(),LEAST_SQUARES,
				   validation_criteria=MSE(),process_labels=false)
model = salsa(X,y,model,Xtest)

@test_approx_eq_eps mse(sinc(Xtest), model.output.Ytest) 0.01 0.01