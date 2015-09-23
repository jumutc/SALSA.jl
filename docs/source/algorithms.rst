Algorithms
==========

This part of the package provides a description, API and references to the implemented core algorithmic schemes (solvers) available in the SALSA package. Every algorithm can be supplied to ``salsa`` subroutines either directly (see :func:`salsa`) or passed within ``SALSAModel``. Another available API is shipped with direct calls to algorithmic schemes. The latter is the most primitive and basic way of using SALSA package.


Available high-level API
~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: PEGASOS

	Defines an implementation of the `Pegasos: Primal Estimated sub-GrAdient SOlver for SVM <http://ttic.uchicago.edu/~shai/papers/ShalevSiSr07.pdf>`_ which solves :math:`l_2`-regularized problem defined `here <index.rst#mathematical background>`__.
	
.. function:: L1RDA
	
	Defines an implementation of the `l1-Regularized Dual Averaging <http://research.microsoft.com/pubs/141578/xiao10JMLR.pdf>`_ solver which solves elastic-net regularized problem defined `here <index.rst#mathematical background>`__.