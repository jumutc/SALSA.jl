export RBFKernel

immutable RBFKernel <: Kernel
    sigma2::Float64

    function RBFKernel(sigma2::Float64)
        if sigma2 <= zero(sigma2)
            error("sigma2 must be positive")
        else
            new(sigma2)
        end
    end
end

function kernel_from_data_model(k::Type{RBFKernel}, X)
    (N,dim) = size(X)
    sig2 = mean((std(X)*(N^(-1/(dim+4)))).^2)
    kernel_from_parameters(k,exp(sig2))
end


function kernel_matrix(k::RBFKernel, X::Matrix)
    n = size(X,1)
    K = X*X'
    dX = diag(K)
    for j=1:n, i=1:n
        K[i,j] = exp( -(dX[i] + dX[j] - 2*K[i,j])/(2*k.sigma2) )
    end
    K
end

function kernel_matrix(k::RBFKernel, Xr::Matrix, Xc::Matrix) 
    nXr = size(Xr, 1)
    nXc = size(Xc, 1)
    K = Xr*Xc'

    XrDotXr = Array(Float64,nXr)
    for i=1:nXr
        v = Xr[i,:]
        XrDotXr[i] = vecnorm(v)^2
    end

    XcDotXc = Array(Float64,nXc)
    for i=1:nXc
        v = Xc[i,:]
        XcDotXc[i] = vecnorm(v)^2
    end

    for j=1:nXc, i=1:nXr
        K[i,j] = exp( -( XrDotXr[i] + XcDotXc[j] - 2*K[i,j] ) / ( 2*k.sigma2 ) )
    end
    K
end