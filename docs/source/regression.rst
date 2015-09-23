Regression
================

A regression example is explained for the SALSA package by the ``sinc(x) = sin(x)./x`` function.


This package provides a function ``salsa`` and explanation on ``SALSAModel`` for the regression case. This use case is supported by the Fixed-Size approach [FS2010]_ and :doc:`Nyström approximation <nystrom>` with the specific ``LEAST_SQUARES`` loss function and cross-validation criterion ``MSE`` (mean-squared error). 

.. code-block:: julia

    using SALSA, Base.Test

    srand(1234)
    sinc(x) = sin(x)./x
    X = linspace(0.1,20,100)''
    Xtest = linspace(0.11,19.9,100)''
    y = sinc(X)

    model = SALSAModel(NONLINEAR, SIMPLE_SGD(), LEAST_SQUARES,
				validation_criterion=MSE(), process_labels=false)
    model = salsa(X, y, model, Xtest)

    @test_approx_eq_eps mse(sinc(Xtest), model.output.Ytest) 0.05 0.01

By taking a look at the code snippet above we can notice a major difference with the :doc:`Classification <classification>` example. The model is equipped with the ``NONLINEAR`` mode, ``LEAST_SQUARES`` loss function while the cross-validation criterion is given by ``MSE``. Another important model-related parameter is ``process_labels`` which should be set to ``false`` in order to switch into regression mode. These four essential components unambiguously define a regression problem solved stochastically by the ``SALSA`` package.     

.. [FS2010] De Brabanter K., De Brabanter J., Suykens J.A.K., De Moor B., "Optimized Fixed-Size Kernel Models for Large Data Sets", Computational Statistics & Data Analysis, vol. 54, no. 6, Jun. 2010, pp. 1484-1504.