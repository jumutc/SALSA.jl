export loss_derivative

function hinge_loss_derivative(At,yt,eval)
  idx = find(eval .< 1)
  if ~isempty(idx) 
    hinge_loss(At,yt,idx) 
  else 
    init(At) 
  end
end

# introduce polymorphism for sparse matrices
function pinball_loss_derivative(At,yt,eval,tau) 
   d = init(At) 
   idx = find(eval .< 1)
   idx_neg = setdiff(1:1:size(At,2),idx)
   if ~isempty(idx) 
       d = -sum(At[:,idx].*repmat(yt[idx]',size(At,1),1),2)
   end
   if ~isempty(idx_neg) 
       d = d .+ tau.*sum(At[:,idx_neg].*repmat(yt[idx_neg]',size(At,1),1),2)
   end
   d
end

pinball_loss_derivative(At::Array{Float64,1},yt,eval,tau) = eval < 1 ? -At.*yt : tau*At.*yt

init(At::MVar) = zeros(size(At,1),1) 
init(At::Array{Float64,1}) = zeros(length(At),1) 
init(At::SparseMatrixCSC) = spzeros(size(At,1),1) 

hinge_loss(At::Array{Float64,1},yt,idx) = -At.*yt
hinge_loss(At::MVar,yt,idx) = -sum(At[:,idx].*repmat(yt[idx]',size(At,1),1),2)
hinge_loss(At::SparseMatrixCSC,yt,idx) = reduce((d0,i) -> d0 - (At[:,i] .* yt[i]), spzeros(size(At,1),1), Set(idx))

logistic_loss(At::Array{Float64,1},yt,eval) = -At.*yt/(exp(eval)+1) 
logistic_loss(At::MVar,yt,eval) = -sum(At.*repmat((yt./(exp(eval)+1))',size(At,1),1),2)
logistic_loss(At::SparseMatrixCSC,yt,eval) = reduce((d0,i) -> d0 - (At[:,i] .* (yt[i]/(exp(eval[i])+1))), spzeros(size(At,1),1), 1:1:size(At,1))

# derivative of the Loss function
loss_derivative(::Type{LOGISTIC}) = logistic_loss
loss_derivative(::Type{HINGE}) = hinge_loss_derivative
loss_derivative(::Type{PINBALL},tau::Float64) = (At,yt,idx) -> pinball_loss_derivative(At,yt,idx,tau)