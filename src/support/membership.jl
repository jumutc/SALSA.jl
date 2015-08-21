function membership(Y::Matrix{Float64})
    max_dists = maximum(Y, 2)[:]

    (x,y) = findn(Y .== max_dists)
    memb_val = zeros(size(Y,1))
    memb_val[x] = y
    memb_val
end