Model Tuning
============

This part of the package provides a simple API for model-tuning routines.

.. function:: gen_cross_validate(evalfun,n,model)

	Perform in parallel a generic cross-validation routine defined in ``evalfun`` by the splitting specified in ``model.cv_gen``.
	
	:param evalfun: function to evaluate
	:param n: total number of data points (instances) if ``model.cv_gen`` is undefined (null)
	:param model: ``SALSAModel`` which contains the ``cv_gen`` field of type ``Nullable{CrossValGenerator}`` (wrapper around the type defined in ``MLBase.jl`` package)
	
	:return: an average of ``evalfun`` evaluations.
	

.. function:: misclass(y,yhat)

	Calculate misclassification rate as :math:`\frac{1}{n}\sum_{i=1}^n I(y-I \neq \hat{y}_i)`.
	
.. function:: mse(y,yhat)

	Calculate mean squared error as :math:`\frac{1}{n}\|y - \hat{y}\|^2`
	
.. function:: auc(y,yhat[,n=100])

	Calculate Area Under `ROC <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>` Curve. Default number of thresholds is 100.