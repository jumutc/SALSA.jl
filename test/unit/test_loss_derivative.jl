using SALSA, Base.Test

loss = loss_derivative(HINGE)
@test loss([1;2],-1,[2;1]) == [1;2]
@test loss([1;2],1,[2;1])  == zeros(2,1)
#The old test was loss([1 1;2 2],[-1,-1],[2;1]) == [10,20]''
#However '' no longer changes an array of length n to a nx1 array
@test loss([1 1;2 2],[-1,-1],[2;1]) == reshape([2,4], :, 1)
@test loss(sparse([1;2]),[-1],sparse([2;1])) == sparse(reshape([1;2], :, 1))

loss = loss_derivative(SQUARED_HINGE)
@test loss([1;2],-1,[2;1]) == [5;10]
@test loss([1;2],1,[2;1])  == zeros(2,1)
#The old test was loss([1 1;2 2],[-1,-1],[2;1]) == [10,20]''
#However '' no longer changes an array of length n to a nx1 array
@test loss([1 1;2 2],[-1,-1],[2;1]) == reshape([10,20], :, 1)
@test loss(sparse([1;2]),[-1],sparse([2;1])) == sparse(reshape([5;10], :, 1))

loss = loss_derivative(PINBALL,.5)
@test loss([1;2],-1,[2;1]) == [1;2]
@test loss([1;2],1,[2;1])  == [1;2].*.5
@test loss([1 2;2 1],[-1,1],[2;1]) == reshape([2,2.5], :, 1)
@test loss(sparse([1;2]),[-1],sparse([2;1])) == sparse(reshape([1;2], :, 1))

loss = loss_derivative(LOGISTIC)
@test loss([1;2],-1,[2;1]) == [0.9820137900379085,1.964027580075817]
@test loss([1;2],1,[2;1])  == [-0.01798620996209156,-0.03597241992418312]
@test loss([1 2;2 1],[-1,1],[2;1]) == reshape([0.9686280881893388,1.957334729151532], :, 1)
@test loss(sparse([1;2]),[-1],sparse([2;1])) == sparse(reshape([0.9820137900379085,1.964027580075817], :, 1))

loss = loss_derivative(LEAST_SQUARES)
@test loss([1;2],-1,[2;1]) == [5,10]
@test loss([1;2],1,[2;1])  == [3,6]
@test loss([1 2;2 1],[-1,1],[2;1]) == [13,14]
@test loss(sparse([1;2]),[-1],sparse([2;1])) == sparse(reshape([5,10], :, 1))

loss = loss_derivative(MODIFIED_HUBER)
@test loss([1;2],-1,[2;1]) == reshape([4,8], :, 1)
@test loss([1;2],1,[.1;.1]) == reshape([-.7,-1.4], :, 1)
@test loss([1 1;2 2],[-1,1],[.1;.2]) == reshape([1,2], :, 1)
