using SALSA, Base.Test

loss = loss_derivative(HINGE)
@test loss([1;2],-1,[2;1]) == [1;2]
@test loss([1;2],1,[2;1])  == zeros(2,1)
@test loss([1 1;2 2],[-1,-1],[2;1]) == [2,4]''
@test loss(sparse([1;2]),[-1],sparse([2;1])) == sparse([1;2])

loss = loss_derivative(SQUARED_HINGE)
@test loss([1;2],-1,[2;1]) == [5;10]
@test loss([1;2],1,[2;1])  == zeros(2,1)
@test loss([1 1;2 2],[-1,-1],[2;1]) == [10,20]''
@test loss(sparse([1;2]),[-1],sparse([2;1])) == sparse([5;10])

loss = loss_derivative(PINBALL,.5)
@test loss([1;2],-1,[2;1]) == [1;2]
@test loss([1;2],1,[2;1])  == [1;2].*.5
@test loss([1 2;2 1],[-1,1],[2;1]) == [2,2.5]''
@test loss(sparse([1;2]),[-1],sparse([2;1])) == sparse([1;2])

loss = loss_derivative(LOGISTIC)
@test loss([1;2],-1,[2;1]) == [0.9820137900379085,1.964027580075817]
@test loss([1;2],1,[2;1])  == [-0.01798620996209156,-0.03597241992418312]
@test loss([1 2;2 1],[-1,1],[2;1]) == [0.9686280881893388,1.957334729151532]''
@test loss(sparse([1;2]),[-1],sparse([2;1])) == sparse([0.9820137900379085,1.964027580075817])

loss = loss_derivative(LEAST_SQUARES)
@test loss([1;2],-1,[2;1]) == [5,10]
@test loss([1;2],1,[2;1])  == [3,6]
@test loss([1 2;2 1],[-1,1],[2;1]) == [13,14]
@test loss(sparse([1;2]),[-1],sparse([2;1])) == sparse([5,10])

loss = loss_derivative(MODIFIED_HUBER)
@test loss([1;2],-1,[2;1]) == [4,8]''
@test loss([1;2],1,[.1;.1]) == [-.7,-1.4]''
@test loss([1 1;2 2],[-1,1],[.1;.2]) == [1,2]''