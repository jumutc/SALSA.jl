Regression
================

A regression example explained by the usage of SALSA package on the ``sinc(x) = sin(x)./x`` function.


This package provides a function ``salsa`` and explanation on ``SALSAModel`` which enables a regression case. This use case is supported by the Fixed-Size approach [FS2008]_ and :doc:`Nyström approximation <nystrom>` with the specific ``LEAST_SQUARES`` loss function and cross-validation criteria ``MSE`` (mean-squared error). 

.. code-block:: julia

    using SALSA, Base.Test

    srand(1234)
    sinc(x) = sin(x)./x
    X = linspace(0.1,20,100)''
    Xtest = linspace(0.11,19.9,100)''
    y = sinc(X)

    model = SALSAModel(NONLINEAR,PEGASOS(),LEAST_SQUARES,validation_criteria=MSE())
    model = salsa(X,y,model,Xtest)

    @test_approx_eq_eps mse(sinc(Xtest), model.output.Ytest) 0.01 0.01

By taking a look at the code snippet above we can notice a major difference with the :doc:`Classification <classification>` example. The model is equipped with the ``NONLINEAR`` mode, ``LEAST_SQUARES`` loss function while the cross-validation criteria is given by ``MSE``. These three essential components unambiguously define a regression problem solved stochastically by the ```SALSA``` package.     

.. [FS2008] De Brabanter K., De Brabanter J., Suykens J.A.K., De Moor B., "Optimized Fixed-Size Kernel Models for Large Data Sets", Computational Statistics & Data Analysis, vol. 54, no. 6, Jun. 2010, pp. 1484-1504.