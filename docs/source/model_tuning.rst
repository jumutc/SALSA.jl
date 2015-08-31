Model Tuning
============

This part of the package provides a simple API for model-tuning routines.

.. function:: gen_cross_validate(evalfun,n,model)

	Perform in parallel a generic cross-validation routine defined in ``evalfun`` by the splitting specified in ``model.cv_gen``.
	
	:param evalfun: function to evaluate
	:param n: total number of data points (instances) if ``model.cv_gen`` is undefined.
	:param model: ``SALSAModel`` which contains ``cv_gen`` of type ``Nullable{CrossValGenerator}`` (wrapper around the type defined in ``MLBase`` package)
	
	:return: an average of ``evalfun`` evaluations 