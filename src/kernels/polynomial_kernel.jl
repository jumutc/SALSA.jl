immutable PolynomialKernel <: Kernel
    tau::Float64
    d::Float64
end

function kernel_matrix(k::PolynomialKernel, Xr::Matrix, Xc::Matrix)
    nXr = size(Xr,1)
    nXc = size(Xc,1)
    K = Xr*Xc'
    for j=1:nXc, i=1:nXr
        K[i,j] = (K[i,j] + k.tau)^ceil(k.d)
    end
    K
end