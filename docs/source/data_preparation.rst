Data Preprocessing 
==================

This part of the package provides a simple set of preprocessing utilities.

Data Normalization
~~~~~~~~~~~~~~~~~~

.. function:: mapstd(X)

    Normalize each column of ``X`` to zero mean and one standard deviation. Output normalized matrix ``X`` with extracted column-wise means and standard deviations.

    .. code-block:: julia

        using SALSA

        mapstd([0 1; -1 2]) # --> ([0.707107  -0.707107; -0.707107   0.707107], [-0.5  1.5], [0.707107 0.707107])


.. function:: mapstd(X,mean,std)

    Normalize each column of ``A`` to the specified column-wise ``mean`` and ``std``. Output normalized matrix ``X``.

    .. code-block:: julia

        using SALSA

        mapstd([0 1; -1 2], [-0.5  1.5], [0.707107 0.707107]) # --> [0.707107  -0.707107; -0.707107   0.707107]


Sparse Data Preparation
~~~~~~~~~~~~~~~~~~~~~~~

.. function::  make_sparse(tuples[,sizes,delim])
    
    Create ``SparseMatrixCSC`` object from matrix of tuples ``Matrix{ASCIIString}`` containing ``index:value`` pairs. The index and value pair can be separated by ``delim`` character, e.g. ``:``. The user can optionally specify final dimensions of the ``SparseMatrixCSC`` object as ``sizes`` tuple.

    :param tuples: matrix of tuples ``Matrix{ASCIIString}`` containing ``index:value`` pairs
    :param sizes: optional tuple of final dimensions, e.g. ``(100000,10)`` (empty by default)
    :param delim: optional character separating index and value pair in each cell of ``tuples``, default is ":"

    :return: ``SparseMatrixCSC`` object.