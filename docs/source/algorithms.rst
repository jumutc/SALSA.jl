Algorithms
==========

This part of the package provides a description, API and references to the implemented core algorithmic schemes (solvers) available in the SALSA package. Every algorithm can be supplied to ``salsa`` subroutines either directly (see :func:`salsa`) or passed within ``SALSAModel``. Another available API is shipped with direct calls to algorithmic schemes. The latter is the most primitive and basic way of using SALSA package.


Available high-level API
~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: PEGASOS

	Defines an implementation of the `Pegasos: Primal Estimated sub-GrAdient SOlver for SVM <http://ttic.uchicago.edu/~shai/papers/ShalevSiSr07.pdf>`_ which solves :math:`l_2`-regularized problem defined :ref:`here <problem_def>`.
	
.. function:: L1RDA
	
	Defines an implementation of the `l1-Regularized Dual Averaging <http://research.microsoft.com/pubs/141578/xiao10JMLR.pdf>`_ solver which solves elastic-net regularized problem defined :ref:`here <problem_def>`.
	
.. function:: ADA_L1RDA

	Defines an implementation of the `Adaptive l1-Regularized Dual Averaging <ttp://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_ solver which solves elastic-net regularized problem defined :ref:`here <problem_def>` in an adaptive way [#f1]_.
	
.. function:: R_L1RDA
	
	Defines an implementation of the `Reweighted l1-Regularized Dual Averaging <ftp://ftp.esat.kuleuven.be/pub/SISTA/vjumutc/reports/reweighted_l1rda_jumutc_suykens.pdf>`_ solver which approximates :math:`l_0`-regularized problem in a limit.
	
.. function:: R_L2RDA
	
	Defines an implementation of the `Reweighted l2-Regularized Dual Averaging <ftp://ftp.esat.kuleuven.be/pub/SISTA/vjumutc/reports/isnn2014_jumutc_suykens.pdf>`_ solver which approximates :math:`l_0`-regularized problem in a limit.
	
.. function:: SIMPLE_SGD

	Defines an implementation of the unconstrained Stochastic Gradient Descent scheme which solves :math:`l_2`-regularized problem defined :ref:`here <problem_def>`.	
	
.. function:: RK_MEANS(support_alg,k_clusters,max_iter,metric)

	Defines an implementation of the Regularized Stochastic K-Means approach [JS2015]_. Please refer to :doc:`Clustering <clustering>` section for examples.
	
	:param support_alg: underlying support algorithm, *e.g.* ``PEGASOS``
	:param k_clusters: number of clusters to locate
	:param max_iter: maximum number of outer iterations
	:param metric: metric to evaluate distances to centroids [#f2]_
	
	Selected ``metric`` unambiguously define a loss function used to learn centroids. Currently supported metrics are:
	
	- ``Euclidean()`` which is complemented by :func:`LEAST_SQUARES` loss function
	- ``CosineDist()`` which is complemented by :func:`HINGE` loss function
	
	
Available low-level API
~~~~~~~~~~~~~~~~~~~~~~~~


.. rubric:: Footnotes
	
.. [#f1] adaptation is taken with respect to observed (sub)gradients of the :doc:`loss function <loss_functions>`
.. [#f2] metric types are defined in `Distances.jl <https://github.com/JuliaStats/Distances.jl>`_ package