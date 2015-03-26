export LinearKernel

type LinearKernel <: Kernel
end

function kernel_matrix(k::LinearKernel, Xr::MVar, Xc::MVar)
    Xr*Xc'
end