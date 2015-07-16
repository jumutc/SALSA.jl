export loss_derivative

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

  idx1 = map(Bool,eval .< 0)
  idx2 = map(Bool,eval .>= 0) & map(Bool,eval .< 1)

  if all(~idx1)
    deriv += 4*hinge_loss(At,yt,idx1)
  end

  if all(~idx2)
    mult = 2*(yt[idx2].*(eval[idx2] - 1))'
    deriv += sum(At[:,idx2].*repmat(mult,size(At,1),1),2)
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

hinge_loss{T <: Number, AM <: AbstractMatrix}(At::AM,yt::T,idx) = -At.*yt
hinge_loss(At::Matrix,yt::Vector,idx) = -sum(At[:,idx].*repmat(yt[idx]',size(At,1),1),2)
hinge_loss(At::SparseMatrixCSC,yt,idx) = reduce((d0,i) -> d0 - (At[:,i] .* yt[i]), spzeros(size(At,1),1), Set(idx))

# PINBALL LOSS
# introduce polymorphism for sparse matrices
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

pinball_loss_derivative{T <: Number, AM <: AbstractMatrix}(At::AM,yt::T,w,tau) = evaluate(At,yt,w) < 1 ? -At.*yt : tau*At.*yt

# LOGISTIC LOSS
logistic_loss(At::Matrix,yt,w,eval=evaluate(At,yt,w)) = -sum(At.*repmat((yt./(exp(eval)+1))',size(At,1),1),2)
logistic_loss{T <: Number, AM <: AbstractMatrix}(At::AM,yt::T,w,eval=evaluate(At,yt,w)) = -At.*yt/(exp(eval)+1) 
logistic_loss(At::SparseMatrixCSC,yt,w,eval=evaluate(At,yt,w)) = reduce((d0,i) -> d0 + (At[:,i] .* (yt[i]/(exp(eval[i])+1))), spzeros(size(At,1),1), 1:1:size(At,2))

# LEAST-SQUARES LOSS
least_squares_loss(At::Matrix,yt,w) = At *(evaluate(At,w) - yt)
least_squares_loss{T <: Number, AM <: AbstractMatrix}(At::AM,yt::T,w) = At.*(evaluate(At,w) - yt)
least_squares_loss(At::SparseMatrixCSC,yt,w) = reduce((d0,i) -> d0 + (At[:,i] .* (sum(At[:,i].*w) - yt[i])), spzeros(size(At,1),1), 1:1:size(At,2))


# aliases of the derivatives for different loss functions
loss_derivative(::Type{LOGISTIC}) = logistic_loss
loss_derivative(::Type{HINGE}) = hinge_loss_derivative
loss_derivative(::Type{LEAST_SQUARES}) = least_squares_loss
loss_derivative(::Type{MODIFIED_HUBER}) = modified_huber_loss_derivative
loss_derivative(::Type{PINBALL},tau::Float64) = (At,yt,w) -> pinball_loss_derivative(At,yt,w,tau)
loss_derivative(::Type{RK_MEANS}) = (At::Matrix,yt,w) ->  reduce((d0,i) -> d0 + (w - At[:,i]), zeros(size(At,1),1), 1:1:size(At,2))