# Solution evaluation at sample(s) At with or w/o labels yt
evaluate(At::AbstractMatrix,yt,w) = map(i->sum(At[:,i].*w),1:1:length(yt)).*yt
evaluate(At::AbstractMatrix,w)    = map(i->sum(At[:,i].*w),1:1:size(At,2))
evaluate(At::Vector,yt,w) = dot(At,w[:])*yt
evaluate(At::Vector,w)    = dot(At,w[:])

init(At::Matrix) = zeros(size(At,1),1) 
init(At::Vector) = zeros(length(At),1) 
init(At::SparseMatrixCSC) = spzeros(size(At,1),1) 

# MODIFIED HUBER LOSS
function modified_huber_loss_derivative(At,yt,w)
  eval = evaluate(At,yt,w)
  deriv = init(At)

  idx1 = find(eval.<-1)
  idx2 = find((eval.>=-1)&(eval.<1))

  if ~isempty(idx1)
    deriv += 4*hinge_loss(At,yt,idx1)
  end

  if ~isempty(idx2)
    deriv += squared_hinge_loss(At,yt,idx2,eval)
  end

  deriv
end

# HINGE LOSS
function hinge_loss_derivative(At,yt,w)
  idx = find(evaluate(At,yt,w) .< 1)
  if ~isempty(idx) 
    hinge_loss(At,yt,idx) 
  else 
    init(At) 
  end
end

hinge_loss{T <: Number}(At,yt::T,idx) = -At.*yt
hinge_loss(At::Matrix,yt::Vector,idx) = -sum(At[:,idx].*repmat(yt[idx]',size(At,1),1),2)
hinge_loss(At::SparseMatrixCSC,yt,idx) = reduce((d0,i) -> d0 - (At[:,i] .* yt[i]), spzeros(size(At,1),1), Set(idx))

# SQUARED HINGE LOSS
function squared_hinge_loss_derivative(At,yt,w)
  eval = evaluate(At,yt,w)
  idx = find(eval .< 1)
  if ~isempty(idx) 
    squared_hinge_loss(At,yt,idx,eval) 
  else 
    init(At) 
  end
end

squared_hinge_loss{T <: Number}(At,yt::T,idx,eval) = -At.*(yt - eval*yt)
squared_hinge_loss(At::Matrix,yt::Vector,idx,eval) = -sum(At[:,idx].*repmat((yt[idx].*(1 - eval[idx]))',size(At,1),1),2)
squared_hinge_loss(At::SparseMatrixCSC,yt,idx,eval) = reduce((d0,i) -> d0 - (At[:,i] .* yt[i]*(1 - eval[i])), spzeros(size(At,1),1), Set(idx))

# PINBALL (quantile) LOSS
function pinball_loss_derivative(At,yt,w,tau) 
   d = init(At) 
   idx = find(evaluate(At,yt,w) .< 1)
   idx_neg = setdiff(1:size(At,2),idx)
   if ~isempty(idx) 
       d = -sum(At[:,idx].*repmat(yt[idx]',size(At,1),1),2)
   end
   if ~isempty(idx_neg) 
       d = d .+ tau.*sum(At[:,idx_neg].*repmat(yt[idx_neg]',size(At,1),1),2)
   end
   d
end

pinball_loss_derivative{T <: Number, AV <: AbstractVector}(At::AV,yt::T,w,τ) = evaluate(At,yt,w) < 1 ? -At.*yt : τ*At.*yt
pinball_loss_derivative{T <: Number}(At::SparseMatrixCSC,yt::T,w,τ) = evaluate(At,yt,w)[1] < 1 ? -At.*yt : τ*At.*yt

# LOGISTIC LOSS
logistic_loss{T <: Number}(At,yt::T,w,eval=evaluate(At,yt,w)) = -At.*yt/(exp(eval)+1) 
logistic_loss(At::Matrix,yt,w,eval=evaluate(At,yt,w)) = -sum(At.*repmat((yt./(exp(eval)+1))',size(At,1),1),2)
logistic_loss(At::SparseMatrixCSC,yt,w,eval=evaluate(At,yt,w)) = reduce((d0,i) -> d0 - (At[:,i] .* (yt[i]/(exp(eval[i])+1))), spzeros(size(At,1),1), 1:1:size(At,2))

# LEAST-SQUARES LOSS
least_squares_loss(At::Matrix,yt,w) = At*(evaluate(At,w) - yt)
least_squares_loss{T <: Number}(At,yt::T,w) = At.*(evaluate(At,w) - yt)
least_squares_loss(At::SparseMatrixCSC,yt,w) = reduce((d0,i) -> d0 + (At[:,i] .* (sum(At[:,i].*w) - yt[i])), spzeros(size(At,1),1), 1:1:size(At,2))

# LOSS functions for clustering
clustering_least_squares_loss(At::AbstractMatrix,yt,w) = reduce((d0,i) -> d0 + (w - At[:,i]), zeros(size(At,1),1), 1:1:size(At,2))
clustering_hinge_loss(At::Matrix,yt,w) = begin idx = find(evaluate(At,yt,w) .<= 0); isempty(idx) ? init(At) : -sum(At[:,idx],2) end
clustering_hinge_loss(At::SparseMatrixCSC,yt,w) = begin idx = find(evaluate(At,yt,w) .<= 0); isempty(idx) ? init(At) : sparse(-sum(At[:,idx],2)) end

# aliases of the derivatives for different loss functions
loss_derivative(::Type{LOGISTIC}) = logistic_loss
loss_derivative(::Type{HINGE}) = hinge_loss_derivative
loss_derivative(::Type{LEAST_SQUARES}) = least_squares_loss
loss_derivative(::Type{SQUARED_HINGE}) = squared_hinge_loss_derivative
loss_derivative(::Type{MODIFIED_HUBER}) = modified_huber_loss_derivative
loss_derivative{A <: Algorithm, M <: CosineDist}(alg::RK_MEANS{A,M}) = clustering_hinge_loss
loss_derivative{A <: Algorithm, M <: Euclidean}(alg::RK_MEANS{A,M}) = clustering_least_squares_loss
loss_derivative(::Type{PINBALL},tau::Float64) = (At,yt,w) -> pinball_loss_derivative(At,yt,w,tau)