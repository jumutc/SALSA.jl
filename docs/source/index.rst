.. image:: ../images/SALSA.png
    :width: 130px

==================================
.. image:: https://travis-ci.org/jumutc/SALSA.jl.svg
    :target: https://travis-ci.org/jumutc/SALSA.jl
    
.. image:: https://coveralls.io/repos/jumutc/SALSA.jl/badge.svg
	:target: https://coveralls.io/r/jumutc/SALSA.jl

.. image:: https://readthedocs.org/projects/salsajl/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: https://readthedocs.org/projects/salsajl/

Welcome to SALSA's documentation!
==================================


**SALSA**: ``Software`` Lab for ``Advanced`` Machine ``Learning`` and ``Stochastic`` ``Algorithms`` is a native Julia implementation under `GPLv3 license <https://github.com/jumutc/SALSA.jl/blob/master/LICENSE>`_ of stochastic algorithms for: 

- linear and non-linear **Support Vector Machines**
- **sparse linear modelling**

|
Mathematical background
************************

The **SALSA** package aims at stochastically learning a classifier or regressor via the Regularized Empirical Risk Minimization [Vapnik1992]_ framework. We approach a family of the well-known Machine Learning problems of the type:

.. math::
        \min_{\bf w} \sum_{i=1}^n \ell({\bf w},\xi_i) + \Omega({\bf w}),

where :math:`\xi_i = ({\bf x}_i,y_i)` is given as a pair of input-output variables and belongs to a set :math:`\mathcal{S} = \{\xi_{t}\}_{1 \leq t \leq n}` of independent observations, the loss functions :math:`\ell({\bf w},\xi_i)` measures the disagreement between the true target :math:`y` and the model prediction :math:`\hat{y}` while the regularization term :math:`\Omega({\bf w})` penalizes the complexity of the model :math:`{\bf w}`. We draw uniformly :math:`\xi_i` from :math:`\mathcal{S}` at most :math:`T` times due of the `i.i.d. <https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables>`_ assumption and a fixed computational budget. The package includes stochastic algorithms for linear and non-linear Support Vector Machines [Boser1992]_ and sparse linear modelling [Hastie2015]_.

Particular choices of loss functions are (but are not restricted to the selection below):

- `hinge loss <https://en.wikipedia.org/wiki/Hinge_loss>`_ 
- `logistic loss <https://en.wikipedia.org/wiki/Loss_functions_for_classification#Logistic_loss>`_
- least squares loss
- etc.

Particular choices of the `regularization term <https://en.wikipedia.org/wiki/Regularization_(mathematics)>`_ are:

- :math:`l_2`-regularization, *i.e.* :math:`\|w\|_2^2`
- `elastic net <https://en.wikipedia.org/wiki/Elastic_net_regularization>`_ regularization, *i.e.*  :math:`\lambda_1\|w\|_1 + \lambda_2\|w\|_2^2`
- reweighted :math:`l_2`-`regularization <ftp://ftp.esat.kuleuven.be/pub/SISTA/vjumutc/reports/isnn2014_jumutc_suykens.pdf>`_
- reweighted :math:`l_1`-`regularization <ftp://ftp.esat.kuleuven.be/pub/SISTA/vjumutc/reports/reweighted_l1rda_jumutc_suykens.pdf>`_

References
***********

**SALSA** is stemmed from the following algorithmic approaches:

- `Pegasos <http://ttic.uchicago.edu/~shai/papers/ShalevSiSr07.pdf>`_: S. Shalev-Shwartz, Y. Singer, N. Srebro, Pegasos: Primal Estimated sub-GrAdient SOlver for SVM, in: Proceedings of the 24th international conference on Machine learning, ICML ’07, New York, NY, USA, 2007, pp. 807–814. 

- `RDA <http://research.microsoft.com/pubs/141578/xiao10JMLR.pdf>`_: L. Xiao, Dual averaging methods for regularized stochastic learning and online optimization, J. Mach. Learn. Res. 11 (2010), pp. 2543–2596. 

- `Adaptive RDA <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_: J. Duchi, E. Hazan, Y. Singer, Adaptive subgradient methods for online learning and stochastic optimization, J. Mach. Learn. Res. 12 (2011), pp. 2121–2159. 

- `Reweighted RDA <ftp://ftp.esat.kuleuven.be/pub/SISTA/vjumutc/reports/isnn2014_jumutc_suykens.pdf>`_: V. Jumutc, J.A.K. Suykens, Reweighted stochastic learning, Neurocomputing Special Issue - ISNN2014, 2015. (In Press) 

Dependencies
*************

- `MLBase <https://github.com/JuliaStats/MLBase.jl>`_: to support generic Machine Learning routines, e.g. cross-validation, performance measures etc.
- `StatsBase <https://github.com/JuliaStats/StatsBase.jl>`_: to support generic routines from Statistics
- `Distances <https://github.com/JuliaStats/Distances.jl>`_: to support distance metrics between vectors
- `Distributions <https://github.com/JuliaStats/Distributions.jl>`_: to support sampling from various distributions
- `DataFrames <https://github.com/JuliaStats/DataFrames.jl>`_: to support and process files instead of in-memory matrices 
- `Clustering <https://github.com/JuliaStats/Clustering.jl>`_: to support Stochastic K-means Clustering (experimental feature)


.. toctree::
   :hidden:
   :maxdepth: 2

   data_preparation
   classification
   regression
   model_tuning
   nystrom
   


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

.. [Vapnik1992] Vapnik, Vladimir. "Principles of risk minimization for learning theory", In Advances in neural information processing systems (NIPS), pp. 831-838. 1992.
.. [Boser1992] Boser, B., Guyon, I., Vapnik, V. "A training algorithm for optimal margin classifiers", In Proceedings of the fifth annual workshop on Computational learning theory - COLT'92., pp. 144-152, 1992.
.. [Hastie2015] Hastie T., Tibshirani R., Wainwright M. Statistical Learning with Sparsity: The Lasso and Generalizations, Chapman & Hall/CRC Monographs on Statistics & Applied Probability, 2015.