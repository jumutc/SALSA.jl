export PolynomialKernel

immutable PolynomialKernel <: Kernel
    tau::Float64
    d::Float64
    function PolynomialKernel(tau, d)
        new(tau, d)
    end
end

function kernel_matrix(k::PolynomialKernel, Xr::MVar, Xc::MVar)
    nXr = size(Xr,1);
    nXc = size(Xc,1);
    K = Xr*Xc';
    for j=1:nXc, i=1:nXr
        K[i,j] = (K[i,j] + k.tau)^k.d
    end
    K
end