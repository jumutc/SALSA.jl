# 
# Software Lab for Advanced Machine Learning with Stochastic Algorithms
# Copyright (c) 2015 Vilen Jumutc, KU Leuven, ESAT-STADIUS 
# License & help @ https://github.com/jumutc/SALSA.jl
# Documentation @ http://salsajl.readthedocs.org
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#

check_printable(value) = typeof(value) <: Array || typeof(value) <: Mode 
print_value(value) = check_printable(value) ? summary(value) : value

show(io::IO, t::PEGASOS) = @printf io "%s (%s)" typeof(t) "Pegasos: Primal Estimated sub-GrAdient SOlver for SVM"
show(io::IO, t::L1RDA) = @printf io "%s (%s)" typeof(t) "l1-Regularized Dual Averaging"
show(io::IO, t::R_L1RDA) = @printf io "%s (%s)" typeof(t) "Reweighted l1-Regularized Dual Averaging"
show(io::IO, t::R_L2RDA) = @printf io "%s (%s)" typeof(t) "Reweighted l2-Regularized Dual Averaging"
show(io::IO, t::ADA_L1RDA) = @printf io "%s (%s)" typeof(t) "Adaptive l1-Regularized Dual Averaging"
show(io::IO, t::DROP_OUT) = @printf io "%s (%s)" typeof(t) "Dropout Pegasos (experimental)"
show(io::IO, t::SIMPLE_SGD) = @printf io "%s (%s)" typeof(t) "Stochastic Gradient Descent"

show(io::IO, t::MSE) = @printf io "%s (%s)" typeof(t) "Mean Squared Error"
show(io::IO, t::MISCLASS) = @printf io "%s (%s)" typeof(t) "Misclassification Rate"
show(io::IO, t::SILHOUETTE) = @printf io "%s (%s)" typeof(t) "Silhouette Index"
show(io::IO, t::AUC) = @printf io "%s (%s with %d thresholds)" typeof(t) "Area Under ROC Curve" t.n_thresholds
show(io::IO, t::CSA) = @printf io "%s (%s)" typeof(t) "Coupled Simulated Annealing"
show(io::IO, t::DS) = @printf io "%s (%s)" typeof(t) "Directional Search"

show(io::IO, t::Type{HINGE}) = @printf io "SALSA.HINGE (%s)" "Hinge Loss, i.e. l(y,p) = max(0,1 - yp)"
show(io::IO, t::Type{LOGISTIC}) = @printf io "SALSA.LOGISTIC (%s)" "Logistic Loss, i.e. l(y,p) = log(1 + exp(-yp))"
show(io::IO, t::Type{LEAST_SQUARES}) = @printf io "SALSA.LEAST_SQUARES (%s)" "Squared Loss, i.e. l(y,p) = 1/2*(p - y)^2"
show(io::IO, t::Type{SQUARED_HINGE}) = @printf io "SALSA.SQUARED_HINGE (%s)" "Squared Hinge Loss, i.e. l(y,p) = max(0,1 - yp)^2"
show(io::IO, t::Type{PINBALL}) = @printf io "SALSA.PINBALL (%s)" "Pinball (quantile) Loss, i.e. l(y,p) = τI(yp>=1)yp + I(yp<1)(1 - yp)"
show(io::IO, t::Type{MODIFIED_HUBER}) = @printf io "SALSA.MODIFIED_HUBER (%s)" "Modified Huber Loss, i.e. l(y,p) = -4I(yp<-1)yp + I(yp>=-1)max(0,1 - yp)^2"

show(io::IO, t::Type{RBFKernel}) = @printf io "SALSA.RBFKernel (%s)" "Radial Basis Function kernel, i.e. k(x,y) = exp(-||x - y||^2/(2σ^2))"
show(io::IO, t::Type{PolynomialKernel}) = @printf io "SALSA.PolynomialKernel (%s)" "Polynomial kernel, i.e. k(x,y) = (<x,y> + τ)^d"
show(io::IO, t::Type{LinearKernel}) = @printf io "SALSA.LinearKernel (%s)" "Linear kernel, i.e. k(x,y) = <x,y>"

function show(io::IO, model::SALSAModel)
    print_with_color(:blue, io, "SALSA model:\n")
    for field in fieldnames(model)
        value = getfield(model,field)
        field == :output ? println() : @printf io "\t%s : %s\n" field print_value(value)
    end
    print_with_color(:blue, io, "SALSA model.output:\n")
    for field in fieldnames(model.output)
        if isdefined(model.output,field) 
            value = getfield(model.output,field)
            @printf io "\t%s : %s\n" field print_value(value)
        end
    end
end

function print_logo()
    r1 = string("\x1b[31m", "_")
    r2 = string("\x1b[31m", "(")
    r3 = string("\x1b[31m", ")")
    g1 = string("\x1b[32m", "_")
    g2 = string("\x1b[32m", "(")
    g3 = string("\x1b[32m", ")")
    w1 = string("\x1b[0m", "_")
    w2 = string("\x1b[0m", "|")
    w3 = string("\x1b[0m", "/")
    version = Pkg.installed("SALSA")

    println("\n
  ____    _    _     ____    _       $r1 $w1   | Software Lab for Advanced Machine Learning
 / ___|  / \\  | |   / ___|  / \\     $r2$r1$r3 $w2  | with Stochastic Algorithms in Julia
 \\___ \\ / _ \\ | |   \\___ \\ / _ \\    | | |  |
  ___) / ___ \\| |___ ___) / ___ \\ $g1 $w2 | |  | Docs \& license: http://salsajl.readthedocs.org
 |____/_/   \\_\\_____|____/_/   \\_$g2$g1$g3$w3 |_|  | CI builds: http://travis-ci.org/jumutc/SALSA.jl
                                  |__/     | Version: $version\n")
end