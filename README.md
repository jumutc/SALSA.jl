<span>
<img src="https://github.com/jumutc/SALSA.jl/blob/master/docs/SALSA.png?raw=true" height="120"></img>
<img src="https://github.com/jumutc/SALSA.jl/blob/master/docs/logo.png?raw=true"></img>
</span>

[![Build Status](https://travis-ci.org/jumutc/SALSA.jl.svg)](https://travis-ci.org/jumutc/SALSA.jl)
[![Coverage Status](https://coveralls.io/repos/jumutc/SALSA.jl/badge.svg)](https://coveralls.io/r/jumutc/SALSA.jl)
[![Documentation Status](https://readthedocs.org/projects/salsajl/badge/?version=latest)](https://readthedocs.org/projects/salsajl/)

## Software Lab
**SALSA**: **S**oftware Lab for **A**dvanced Machine **L**earning with **S**tochastic **A**lgorithms is a native Julia implementation of the well known stochastic algorithms for **sparse linear modelling**, linear and non-linear **Support Vector Machines**. It is stemmed from the following algorithmic approaches:

- [**Pegasos**](http://ttic.uchicago.edu/~shai/papers/ShalevSiSr07.pdf): S. Shalev-Shwartz, Y. Singer, N. Srebro, Pegasos: Primal Estimated sub-GrAdient SOlver for SVM, in: Proceedings of the 24th international conference on Machine learning, ICML ’07, New York, NY, USA, 2007, pp. 807–814. 

- [**RDA**](http://research.microsoft.com/pubs/141578/xiao10JMLR.pdf): L. Xiao, Dual averaging methods for regularized stochastic learning and online optimization, J. Mach. Learn. Res. 11 (2010), pp. 2543–2596. 

- [**Adaptive RDA**](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf): J. Duchi, E. Hazan, Y. Singer, Adaptive subgradient methods for online learning and stochastic optimization, J. Mach. Learn. Res. 12 (2011), pp. 2121–2159. 

- [**Reweighted RDA**](ftp.esat.kuleuven.be/pub/SISTA/vjumutc/reports/reweighted_l1rda_jumutc_suykens.pdf): V. Jumutc, J.A.K. Suykens, Reweighted stochastic learning, Neurocomputing Special Issue - ISNN2014, 2015. (In Press)


## Installation
 - ```Pkg.add("SALSA")```

## Resources
- **Documentation:** <http://salsajl.readthedocs.org>

## Knowledge agnostic usage
```julia
using MAT, SALSA

# Load Ripley data
data = matread(joinpath(Pkg.dir("SALSA"),"data","ripley.mat"))

# Train and cross-validate Pegasos algorithm (default) on training data  
# and evaluate it on the test data provided as the last function argument
model = salsa(data["X"], data["Y"], data["Xt"])

# Compute accuracy in %
@printf "Accuracy: %.2f%%\n" mean(model.output.Ytest .== data["Yt"])*100

# Or use map_predict function and map data beforehand by the extracted mean/std (default) 
@printf "Accuracy: %.2f%%\n" mean(map_predict(model, data["Xt"]) .== data["Yt"])*100
```
or using Q&A tables
```julia
using SALSA

model = salsa_qa(readcsv(joinpath(Pkg.dir("SALSA"),"data","iris.data.csv")))

Do you have any target variable of interest in X (or ENTER for default 'yes')? [y/n]: 

Please provide the column number of your target variable (or ENTER for default last column): 

Is your problem of the classification type (or ENTER for default 'yes')? [y/n]: 

Please select a loss function from options (or ENTER for default)
 	1 : SALSA.PINBALL (Pinball (quantile) Loss, i.e. l(y,p) = τI(yp>=1)yp + I(yp<1)(1 - yp))
	2 : SALSA.HINGE (Hinge Loss, i.e. l(y,p) = max(0,1 - yp)) (default)
	3 : SALSA.LEAST_SQUARES (Squared Loss, i.e. l(y,p) = 1/2*(p - y)^2)
	4 : SALSA.LOGISTIC (Logistic Loss, i.e. l(y,p) = log(1 + exp(-yp)))
	5 : SALSA.MODIFIED_HUBER (Modified Huber Loss, i.e. l(y,p) = -4I(yp<-1)yp + I(yp>=-1)max(0,1 - yp)^2)
	6 : SALSA.SQUARED_HINGE (Squared Hinge Loss, i.e. l(y,p) = max(0,1 - yp)^2)
: 

Please select a cross-validation (CV) criterion from options (or ENTER for default)
 	1 : SALSA.AUC (Area Under ROC Curve with 100 thresholds)
	2 : SALSA.MISCLASS (Misclassification Rate) (default)
	3 : SALSA.MSE (Mean Squared Error)
: 

Do you want to perform Nyström (nonlinear) approximation (or ENTER for default)? [y/n]
 	n : SALSA.LINEAR (default)
	y : SALSA.NONLINEAR
: 

Please select an algorithm from options (or ENTER for default)
 	1 : SALSA.DROP_OUT (Dropout Pegasos (experimental))
	2 : SALSA.PEGASOS (Pegasos: Primal Estimated sub-GrAdient SOlver for SVM) (default)
	3 : SALSA.SIMPLE_SGD (Stochastic Gradient Descent)
	4 : SALSA.ADA_L1RDA (Adaptive l1-Regularized Dual Averaging)
	5 : SALSA.L1RDA (l1-Regularized Dual Averaging)
	6 : SALSA.R_L1RDA (Reweighted l1-Regularized Dual Averaging)
	7 : SALSA.R_L2RDA (Reweighted l2-Regularized Dual Averaging)
: 

Please select a global optimization method from options (or ENTER for default)
 	1 : SALSA.CSA (Coupled Simulated Annealing) (default)
	2 : SALSA.DS (Directional Search)
: 

Computing the model...
```
