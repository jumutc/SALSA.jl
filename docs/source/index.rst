
Welcome to SALSA's documentation!
==================================

.. image:: ../images/SALSA.png
    :align: left
    :width: 59px


**SALSA**: Software Lab for Advanced Machine Learning and Stochastic Algorithms is a native Julia implementation of the well known stochastic algorithms for linear and non-linear **Support Vector Machines**. 

References:
***********

**SALSA** is stemmed from the following algorithmic approaches:

- `Pegasos <http://ttic.uchicago.edu/~shai/papers/ShalevSiSr07.pdf>`_: S. Shalev-Shwartz, Y. Singer, N. Srebro, Pegasos: Primal Estimated sub-GrAdient SOlver for SVM, in: Proceedings of the 24th international conference on Machine learning, ICML ’07, New York, NY, USA, 2007, pp. 807–814. 

- `RDA <http://research.microsoft.com/pubs/141578/xiao10JMLR.pdf>`_: L. Xiao, Dual averaging methods for regularized stochastic learning and online optimization, J. Mach. Learn. Res. 11 (2010), pp. 2543–2596. 

- `Adaptive RDA <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_: J. Duchi, E. Hazan, Y. Singer, Adaptive subgradient methods for online learning and stochastic optimization, J. Mach. Learn. Res. 12 (2011), pp. 2121–2159. 

- `Reweighted RDA <ftp://ftp.esat.kuleuven.be/pub/SISTA/vjumutc/reports/reweighted_l1rda_jumutc_suykens.pdf>`_: V. Jumutc, J. A. K. Suykens, Reweighted l1 dual averaging approach for sparse stochastic learning, in: 22th European Symposium on Artificial Neural Networks, ESANN 2014, Bruges, Belgium, April 23-25, 2014.

Dependencies:
*************

- `MLBase <https://github.com/JuliaStats/MLBase.jl>`_: to support generic Machine Learning routines, e.g. cross-validation, performance measures etc.
- `StatsBase <https://github.com/JuliaStats/StatsBase.jl>`_: to support generic routines from Statistics
- `Distances <https://github.com/JuliaStats/Distances.jl>`_: to support distance metrics between vectors
- `Distributions <https://github.com/JuliaStats/Distributions.jl>`_: to support sampling from various distributions
- `DataFrames <https://github.com/JuliaStats/DataFrames.jl>`_: to support and process files instead of in-memory matrices 
- `Clustering <https://github.com/JuliaStats/Clustering.jl>`_: to support Stochastic K-means Clustering


Contents:
*********

.. toctree::
   :maxdepth: 2

   data_preparation.rst
   classification.rst


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

