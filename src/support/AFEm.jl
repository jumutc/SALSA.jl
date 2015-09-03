AFEm(Xs, k::Kernel, X::DelimitedFile) = AFEm(Xs,k,sub(X,:,:))

function AFEm(Xs, k::Kernel, X)
    #
    # Automatic Feature Extraction by Nystrom method
    #
    # 
    (eigvals, eigvec) = eig_AFEm(Xs, k)
    AFEm(eigvals, eigvec, Xs, k, X)
end

function AFEm(eigvals, eigvec, Xs, k::Kernel, X)
    #
    # Automatic Feature Extraction by Nystrom method
    #
    features = kernel_matrix(k, X, Xs) * eigvec
    features .* repmat(1./sqrt(eigvals'),size(X,1),1)
end

function eig_AFEm(Xs, k::Kernel)
    # eigenvalue decomposition to do...
    omega = kernel_matrix(k, Xs)
    (eigvals, eigvec) = eig(omega + 2.*eye(size(Xs,1)))
    indices = find(x -> x > eps(), eigvals)
    eigvals[indices], eigvec[:,indices]
end