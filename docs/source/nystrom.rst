Nyström Approximation
=====================

While linear techniques operating in the primal (input) space are able to achieve good generalization capabilities in some specific application areas, one cannot in general approximate with the linear model more complex or highly nonlinear functions. We apply a Fixed-Size approach [FS2010]_ and Nyström approximation [WS2001]_ to approximate a kernel-induced feature map with some higher dimensional explicit and approximate feature vector.

We select prototype vectors (small working sample of size :math:`m \ll n`) and construct, for instance an RBF kernel matrix :math:`K` with

.. math::
	K_{ij} = e^{-\frac{\Vert x_i-x_j \Vert ^2}{2\sigma^2} }.

By following the approach in [WS2001]_ an expression for the entries of the approximated feature map :math:`\hat{\Phi}(x) : \mathbb{R}^d \rightarrow \mathbb{R}^m`, with :math:`\hat{\Phi}(x) = (\hat{\Phi}_1(x),\ldots,\hat{\Phi}_m(x))^T` is given by

.. math::
	\hat{\Phi}_i(x) = \frac{1}{\sqrt{\lambda_{i,m}}} \sum_{t=1}^m u_{ti,m}k(x_t,x),

where :math:`\lambda_{i,m}` and :math:`u_{i,m}` denote the *i*-th eigenvalue and the *i*-th eigenvector of :math:`K`.


Avaliable API
~~~~~~~~~~~~~

.. function:: AFEm(Xs,kernel,X)
	
	Perform Automatic Feature Extraction (AFE) by Nyström method [WS2001]_ using a subsample :math:`X_s \in X`. We restrict ``kernel <: Kernel`` to be a subclass of ``Kernel``, for instance ``RBFKernel``.
	
	:param Xs: subsample which is used to construct kernel matrix :math:`K`
	:param kernel: kernel function used to construct kernel matrix :math:`K`
	:param X: full dataset
	
	:return: new dataset :math:`X_f` derived from stacking together feature maps for every :math:`x_i \in X`

.. [FS2010] De Brabanter K., De Brabanter J., Suykens J.A.K., De Moor B., "Optimized Fixed-Size Kernel Models for Large Data Sets", Computational Statistics & Data Analysis, vol. 54, no. 6, Jun. 2010, pp. 1484-1504.
.. [WS2001] Williams C. and Seeger M., "Using the Nyström method to speed up kernel machines", in Proceedings of the 14th Annual Conference on Neural Information Processing (NIPS), pp. 682-688, 2001.