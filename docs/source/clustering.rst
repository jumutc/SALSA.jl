Clustering
================

A clustering example is explained for the SALSA package on the Iris dataset [UCI2010]_. 

This package provides a function ``salsa`` and explanation on ``SALSAModel`` for the clustering case. This use case is supported by the particular choices of loss functions and distance metrics applied within the Regularized K-Means approach [JS2015]_ and cross-validation criterion ``SILHOUETTE`` (`Silhouette index <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_). 

.. code-block:: julia

    using SALSA, Clustering, Distances, MLBase, Base.Test

    Xf = readcsv(joinpath(Pkg.dir("SALSA"), "data", "iris.data.csv"))
	Y = convert(Array{Int}, Xf[:,end])
	dY = Array{Int}(length(Y))
	X = Xf[:,1:end-1]
	max_iter = 20

	srand(1234)
	algorithm = RK_MEANS(max_iter)
	model = SALSAModel(LINEAR, algorithm, LEAST_SQUARES,
				validation_criterion=SILHOUETTE(),
				global_opt=DS([-1]), process_labels=false,
				cv_gen = Nullable{CrossValGenerator}(Kfold(length(Y),3)))
	model = salsa(X, dY, model, X)
	mappings = model.output.Ytest

By taking a close look at the code snippet above we can notice that we use a special type of an algorithm ``RK_MEANS`` which implements approach in [JS2015]_. By instantiating ``RK_MEANS(max_iter)`` we provide a maximum number of outer iterations. Learning of individual prototype vectors will be repeated ```max_iter``` times after re-partitioning of the dataset ``X``. The default choice of the loss function is ``LEAST_SQUARES`` and the distance metric is ``Euclidean()`` [#f1]_. This corresponds to the original setting of the unregularized K-Means approach. Please refer to :doc:`Algorithms <algorithms>` section and :func:`RK_MEANS` function for more details regarding which combinations of loss functions and metrics are supported.

.. [UCI2010] Lichman, M. (2013). `UCI Machine Learning Repository <http://archive.ics.uci.edu/ml>`_. Irvine, CA: University of California, School of Information and Computer Science.
.. [JS2015] Jumutc V., Suykens J.A.K., "Regularized and Sparse Stochastic K-Means for Distributed Large-Scale Clustering", Internal Report 15-126, ESAT-SISTA, KU Leuven (Leuven, Belgium), 2015.

.. rubric:: Footnotes
	
.. [#f1] metric types are defined in `Distances.jl <https://github.com/JuliaStats/Distances.jl>`_ package