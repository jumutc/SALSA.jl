Loss Functions
==============

This part of the package provides a description and mathematical background of the implemented loss functions. Every loss function can be supplied to ``salsa`` subroutines either directly (see :func:`salsa`:) or passed within ``SALSAModel``. In the definitions below 

.. function:: HINGE
	
	Defines an implementation of the `Hinge Loss <https://en.wikipedia.org/wiki/Hinge_loss>`_ function, *i.e.* :math:`l(y,p) = \max(0,1 - yp)`.
	
.. function:: LOGISTIC