Algorithms
==========

This part of the package provides a description, API and references to the implemented core algorithmic schemes (solvers) available in the SALSA package. Every algorithm can be supplied to ``salsa`` subroutines either directly (see :func:`salsa`) or passed within ``SALSAModel``. Another available API is shipped with direct calls to algorithmic schemes. The latter is the most primitive and basic way of using SALSA package.


Available high-level API
~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: PEGASOS

	Defines an implementation (see :func`pegasos_alg`) of the `Pegasos: Primal Estimated sub-GrAdient SOlver for SVM <http://ttic.uchicago.edu/~shai/papers/ShalevSiSr07.pdf>`_ which solves :math:`l_2`-regularized problem defined :ref:`here <problem_def>`.
	
.. function:: L1RDA
	
	Defines an implementation (see :func`l1rda_alg`) of the `l1-Regularized Dual Averaging <http://research.microsoft.com/pubs/141578/xiao10JMLR.pdf>`_ solver which solves elastic-net regularized problem defined :ref:`here <problem_def>`.
	
.. function:: ADA_L1RDA

	Defines an implementation (see :func`adaptive_l1rda_alg`) of the `Adaptive l1-Regularized Dual Averaging <ttp://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_ solver which solves elastic-net regularized problem defined :ref:`here <problem_def>` in an adaptive way [#f1]_.
	
.. function:: R_L1RDA
	
	Defines an implementation (see :func`reweighted_l1rda_alg`) of the `Reweighted l1-Regularized Dual Averaging <ftp://ftp.esat.kuleuven.be/pub/SISTA/vjumutc/reports/reweighted_l1rda_jumutc_suykens.pdf>`_ solver which approximates :math:`l_0`-regularized problem in a limit.
	
.. function:: R_L2RDA
	
	Defines an implementation (see :func`reweighted_l2rda_alg`) of the `Reweighted l2-Regularized Dual Averaging <ftp://ftp.esat.kuleuven.be/pub/SISTA/vjumutc/reports/isnn2014_jumutc_suykens.pdf>`_ solver which approximates :math:`l_0`-regularized problem in a limit.
	
.. function:: SIMPLE_SGD

	Defines an implementation (see :func`sgd_alg`) of the unconstrained Stochastic Gradient Descent scheme which solves :math:`l_2`-regularized problem defined :ref:`here <problem_def>`.	
	
.. function:: RK_MEANS(support_alg, k_clusters, max_iter, metric)

	Defines an implementation (see :func`stochastic_rk_means`) of the Regularized Stochastic K-Means approach [JS2015]_. Please refer to :doc:`Clustering <clustering>` section for examples.
	
	:param support_alg: underlying support algorithm, *e.g.* ``PEGASOS``
	:param k_clusters: number of clusters to locate
	:param max_iter: maximum number of outer iterations
	:param metric: metric to evaluate distances to centroids [#f2]_
	
	Selected ``metric`` unambiguously define a loss function used to learn centroids. Currently supported metrics are:
	
	- ``Euclidean()`` which is complemented by :func:`LEAST_SQUARES` loss function
	- ``CosineDist()`` which is complemented by :func:`HINGE` loss function
	
	
Available low-level API
~~~~~~~~~~~~~~~~~~~~~~~

.. function:: pegasos_alg(dfunc, X, Y, λ, k, max_iter, tolerance[, online_pass=0, train_idx=[]])

	:param dfunc: supplied loss function derivative (see :func:`loss_derivative`)
	:param X: a full dataset (samples are stacked row-wise) represented by ``Matrix``, ``SparseMatrixCSC`` or :func:`DelimitedFile`
	:param Y: labels corresponding to ``X``
	:param λ: trade-off hyperparameter
	:param k: sampling size at each iteration :math:`t`
	:param max_iter: maximum number of iterations (budget)
	:param tolerance: early stopping threshold, *i.e.* :math:`||w_{t+1} - w_t|| <= tolerance`
	:param online_pass: number of online passes through data, ``online_pass=0`` indicates a default stochastic mode instead of an online mode
	:param train_idx: subset of indices from ``X`` used to learn a model (:math:`w, b`)
	
	:return: :math:`w, b`
	
.. function:: sgd_alg(dfunc, X, Y, λ, k, max_iter, tolerance[, online_pass=0, train_idx=[]])

	:param dfunc: supplied loss function derivative (see :func:`loss_derivative`)
	:param X: a full dataset (samples are stacked row-wise) represented by ``Matrix``, ``SparseMatrixCSC`` or :func:`DelimitedFile`
	:param Y: labels corresponding to ``X``
	:param λ: trade-off hyperparameter
	:param k: sampling size at each iteration :math:`t`
	:param max_iter: maximum number of iterations (budget)
	:param tolerance: early stopping threshold, *i.e.* :math:`||w_{t+1} - w_t|| <= tolerance`
	:param online_pass: number of online passes through data, ``online_pass=0`` indicates a default stochastic mode instead of an online mode
	:param train_idx: subset of indices from ``X`` used to learn a model (:math:`w, b`)
	
	:return: :math:`w, b`
	
.. function:: l1rda_alg(dfunc, X, Y, λ, γ, ρ, k, max_iter, tolerance[, online_pass=0, train_idx=[]])

	:param dfunc: supplied loss function derivative (see :func:`loss_derivative`)
	:param X: a full dataset (samples are stacked row-wise) represented by ``Matrix``, ``SparseMatrixCSC`` or :func:`DelimitedFile`
	:param Y: labels corresponding to ``X``
	:param λ: trade-off hyperparameter
	:param γ: hyperparameter involved in elastic-net regularization
	:param ρ: hyperparameter involved in elastic-net regularization
	:param k: sampling size at each iteration :math:`t`
	:param max_iter: maximum number of iterations (budget)
	:param tolerance: early stopping threshold, *i.e.* :math:`||w_{t+1} - w_t|| <= tolerance`
	:param online_pass: number of online passes through data, ``online_pass=0`` indicates a default stochastic mode instead of an online mode
	:param train_idx: subset of indices from ``X`` used to learn a model (:math:`w, b`)
	
	:return: :math:`w, b`

.. function:: adaptive_l1rda_alg(dfunc, X, Y, λ, γ, ρ, k, max_iter, tolerance[, online_pass=0, train_idx=[]])

	:param dfunc: supplied loss function derivative (see :func:`loss_derivative`)
	:param X: a full dataset (samples are stacked row-wise) represented by ``Matrix``, ``SparseMatrixCSC`` or :func:`DelimitedFile`
	:param Y: labels corresponding to ``X``
	:param λ: trade-off hyperparameter
	:param γ: hyperparameter involved in elastic-net regularization
	:param ρ: hyperparameter involved in elastic-net regularization
	:param k: sampling size at each iteration :math:`t`
	:param max_iter: maximum number of iterations (budget)
	:param tolerance: early stopping threshold, *i.e.* :math:`||w_{t+1} - w_t|| <= tolerance`
	:param online_pass: number of online passes through data, ``online_pass=0`` indicates a default stochastic mode instead of an online mode
	:param train_idx: subset of indices from ``X`` used to learn a model (:math:`w, b`)
	
	:return: :math:`w, b`
	
.. function:: reweighted_l1rda_alg(dfunc, X, Y, λ, γ, ρ, ɛ, max_iter, tolerance[, online_pass=0, train_idx=[]])

	:param dfunc: supplied loss function derivative (see :func:`loss_derivative`)
	:param X: a full dataset (samples are stacked row-wise) represented by ``Matrix``, ``SparseMatrixCSC`` or :func:`DelimitedFile`
	:param Y: labels corresponding to ``X``
	:param λ: trade-off hyperparameter
	:param γ: hyperparameter involved in reweighted formulation of a regularization term
	:param ρ: hyperparameter involved in reweighted formulation of a regularization term
	:param ɛ: reweighting hyperparameter
	:param k: sampling size at each iteration :math:`t`
	:param max_iter: maximum number of iterations (budget)
	:param tolerance: early stopping threshold, *i.e.* :math:`||w_{t+1} - w_t|| <= tolerance`
	:param online_pass: number of online passes through data, ``online_pass=0`` indicates a default stochastic mode instead of an online mode
	:param train_idx: subset of indices from ``X`` used to learn a model (:math:`w, b`)
	
	:return: :math:`w, b`

.. function:: reweighted_l2rda_alg(dfunc, X, Y, λ, ɛ, varɛ, max_iter, tolerance[, online_pass=0, train_idx=[]])

	:param dfunc: supplied loss function derivative (see :func:`loss_derivative`)
	:param X: a full dataset (samples are stacked row-wise) represented by ``Matrix``, ``SparseMatrixCSC`` or :func:`DelimitedFile`
	:param Y: labels corresponding to ``X``
	:param λ: trade-off hyperparameter
	:param ɛ: reweighting hyperparameter
	:param varɛ: sparsification hyperparameter
	:param k: sampling size at each iteration :math:`t`
	:param max_iter: maximum number of iterations (budget)
	:param tolerance: early stopping threshold, *i.e.* :math:`||w_{t+1} - w_t|| <= tolerance`
	:param online_pass: number of online passes through data, ``online_pass=0`` indicates a default stochastic mode instead of an online mode
	:param train_idx: subset of indices from ``X`` used to learn a model (:math:`w, b`)
	
	:return: :math:`w, b`

.. function:: stochastic_rk_means(X, rk_means, alg_params, max_iter, tolerance[, online_pass=0, train_idx=[]])

	:param X: a full dataset (samples are stacked row-wise) represented by ``Matrix``, ``SparseMatrixCSC`` or :func:`DelimitedFile`
	:param rk_means: algorithm defined by :func:`RK_MEANS`
	:param alg_params: hyperparameter of the supporting algorithm in ``rk_means.support_alg``
	:param k: sampling size at each iteration :math:`t`
	:param max_iter: maximum number of iterations (budget)
	:param tolerance: early stopping threshold, *i.e.* :math:`||w_{t+1} - w_t|| <= tolerance`
	:param online_pass: number of online passes through data, ``online_pass=0`` indicates a default stochastic mode instead of an online mode
	:param train_idx: subset of indices from ``X`` used to learn a model (:math:`w, b`)
	
	:return: :math:`w, b`


.. rubric:: Footnotes
	
.. [#f1] adaptation is taken with respect to observed (sub)gradients of the :doc:`loss function <loss_functions>`
.. [#f2] metric types are defined in `Distances.jl <https://github.com/JuliaStats/Distances.jl>`_ package