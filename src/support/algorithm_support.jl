function fix_space(train_idx, N::Int)
    if ~isempty(train_idx)
        (train_idx, size(train_idx,1)) 
    else
        (1:1:N, N)
    end
end

function fix_sampling(online_pass, N::Int)
    if online_pass > 0
        (t,k) -> begin
            s = t % N 
            s > 0 ? s : N
        end
    else
        pd = Categorical(N)
        (t,k) -> rand(pd,k)
    end
end

function fix_iter(online_pass, N::Int, max_iter::Int)
    online_pass > 0 ? N*online_pass : max_iter
end