Nyström Approximation
=====================

While linear techniques operating in the primal (input) space are able to achieve good generalization capabilities in some specific application areas, one cannot in general approximate with the linear model more complex phenomenon and highly nonlinear functions. To overcome restrictions of Algorithm \ref{pegasos_pbl} which operates only in the primal space we apply a Fixed-Size approach [FS2008]_ and Nyström approximation [WS2001]_ to approximate a kernel-induced feature map with some higher dimensional explicit and approximate feature vector.

We select prototype vectors (small working sample of size :math:`m \ll n`) and construct, for instance, an RBF kernel matrix :math:`K` with

.. math::
	K_{ij} = e^{-\frac{\Vert x_i-x_j \Vert ^2}{2\sigma^2} }.

Following approach in [WS2001]_ an expression for the entries of the approximated feature map :math:`\hat{\Phi}(x) : \mathbb{R}^d \rightarrow \mathbb{R}^m`, with :math:`\hat{\Phi}(x) = (\hat{\Phi}_1(x),\ldots,\hat{\Phi}_m(x))^T` is given by

.. math::
	\hat{\Phi}_i(x) = \frac{1}{\sqrt{\lambda_{i,m}}} \sum_{t=1}^m u_{ti,m}k(x_t,x),

where :math:`\lambda_{i,m}` and :math:`u_{i,m}1 denote the :math:`i`-th eigenvalue and the :math:`i1-th eigenvector of :math:`K`.


Avaliable API
~~~~~~~~~~~~~



.. [FS2008] De Brabanter K., De Brabanter J., Suykens J.A.K., De Moor B., "Optimized Fixed-Size Kernel Models for Large Data Sets", Computational Statistics & Data Analysis, vol. 54, no. 6, Jun. 2010, pp. 1484-1504.
.. [WS2001] Williams C. and Seeger M., "Using the Nyström method to speed up kernel machines", in Proceedings of the 14th Annual Conference on Neural Information Processing (NIPS), pp. 682-688, 2001.