function membership(Y::Matrix{Float64})
    max_dists = maximum(Y, 2)[:]

    (x,y) = findn(Y .== max_dists)
    memb_val = Array{Int}(size(Y,1))
    memb_val[x] = y
    memb_val
end