export LinearKernel

type LinearKernel <: Kernel
end

function kernel_matrix(k::LinearKernel, Xr::Matrix, Xc::Matrix)
    Xr*Xc'
end