Classification
================

A classification example explained by the usage of SALSA package on `Ripley <http://www.esat.kuleuven.be/sista/lssvmlab/tutorial/node14.html>`_ data set.

This package provides a function ``salsa`` and explanation on ``SALSAModel`` which accompanies and complements it.

.. function:: salsa(X,Y[,Xtest])

    Create a linear classification model:
    
    .. math::
        \hat{y} = \mathrm{sign}(\langle x, w \rangle + b) 

    based on data given in ``X`` and labeling specified in ``Y``. Optionally evaluate it on ``Xtest``. Data should be given in row-wise format (one sample per row). The classification model is embedded into returned ``model`` as ``model.output``. The choise of different algorithms, loss functions and modes will be explained further in this chapter. 

    .. code-block:: julia

        using SALSA, MAT, Base.Test

        ripley = matread(joinpath(Pkg.dir("SALSA"),"data","ripley.mat"))
        model = salsa(ripley["X"],ripley["Y"],ripley["Xt"]) # --> SALSAModel(...)
        @test_approx_eq_eps mean(ripley["Yt"] .== model.output.Ytest) 0.89 0.01


.. function:: salsa(mode,algorithm,loss,X,Y,Xtest)

    Create a classification model with specified choice of algorithm, mode and loss function.

    :param mode: ``LINEAR`` vs. ``NONLINEAR`` mode specifies whether to use a simple linear classification model or to apply Nyström method for approximating feature map before.
    :param alorithm: stochastic algorithm to learn a classification model, e.g. ``PEGASOS``, ``L1RDA`` etc.
    :param loss: loss function to use when learning a classification model, e.g.  ``HINGE``, ``LOGISTIC`` etc.
    :param X: training data (samples)
    :param Y: training labels
    :param Xtest: test data for out-of-sample evaluation. 

    :return: ``SALSAModel`` object.

    .. code-block:: julia

        using SALSA, MAT, Base.Test

        ripley = matread(joinpath(Pkg.dir("SALSA"),"data","ripley.mat"))
        model = salsa(LINEAR,PEGASOS,HINGE,ripley["X"],ripley["Y"],ripley["Xt"])
        @test_approx_eq_eps mean(ripley["Yt"] .== model.output.Ytest) 0.89 0.01
       