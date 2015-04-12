<img src="http://dl.dropboxusercontent.com/s/ue01x17cs51y9mb/salsa.jpg" width="150"></img>

# SALSA.jl
[![Build Status](https://travis-ci.org/jumutc/SALSA.jl.svg)](https://travis-ci.org/jumutc/SALSA.jl)
[![Coverage Status](https://coveralls.io/repos/jumutc/SALSA.jl/badge.svg)](https://coveralls.io/r/jumutc/SALSA.jl)

## Software Lab
**SALSA**: **S**oftware Lab for **A**dvanced Machine **L**earning and **S**tochastic **A**lgorithms is a native Julia implementation of the well known stochastic algorithms for linear and non-linear **Support Vector Machines**. It is stemmed from the following algorithmic approaches:

- [**Pegasos**](http://ttic.uchicago.edu/~shai/papers/ShalevSiSr07.pdf): S. Shalev-Shwartz, Y. Singer, N. Srebro, Pegasos: Primal Estimated sub-GrAdient SOlver for SVM, in: Proceedings of the 24th international conference on Machine learning, ICML ’07, New York, NY, USA, 2007, pp. 807–814. 

- [**RDA**](http://research.microsoft.com/pubs/141578/xiao10JMLR.pdf): L. Xiao, Dual averaging methods for regularized stochastic learning and online optimization, J. Mach. Learn. Res. 11 (2010), pp. 2543–2596. 

- [**Adaptive RDA**](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf): J. Duchi, E. Hazan, Y. Singer, Adaptive subgradient methods for online learning and stochastic optimization, J. Mach. Learn. Res. 12 (2011), pp. 2121–2159. 

- [**Reweighted RDA**](ftp.esat.kuleuven.be/pub/SISTA/vjumutc/reports/reweighted_l1rda_jumutc_suykens.pdf): V. Jumutc, J. A. K. Suykens, Reweighted l1 dual averaging approach for sparse stochastic learning, in: 22th European Symposium on Artificial Neural Networks, ESANN 2014, Bruges, Belgium, April 23-25, 2014.


## Installation within Julia interpreter
 - ```Pkg.clone("https://github.com/jumutc/SALSA.jl.git")```
 - ```run(`ln -s $(pwd()) $(Pkg.dir("SALSA"))`); Pkg.resolve()``` (from the unzipped source folder)

## Knowledge agnostic usage
```julia
using MAT, SALSA

# Load Ripley data
data = matread(joinpath(Pkg.dir("SALSA"),"data","ripley.mat"))

# Train and cross-validate Pegasos algorithm (default) on training data  
# and evaluate it on the test data provided as the last function argument
model = salsa(data["X"], data["Y"], data["Xt"]);

# Compute accuracy in %
@printf "Accuracy: %.2f%%\n" mean(model.Ytest .== data["Yt"])*100

# Or use map_predict function and map data beforehand by the extracted mean/std (default) 
@printf "Accuracy: %.2f%%\n" mean(map_predict(model, data["Xt"]) .== data["Yt"])*100
```