using SALSA, Base.Test

data = ["1:2.3" "200:4"; "2:2" "100:5.6"]
X = make_sparse(data,delim=":")

@test size(X,1) == 2
@test size(X,2) == 200
@test length(X.nzval) == 4
@test X[2,100] == 5.6

data = ["1" "2.3" "200" "4"; "2" "2" "100" "5.6"]
X = make_sparse(data)

@test size(X,1) == 2
@test size(X,2) == 200
@test length(X.nzval) == 4
@test X[2,100] == 5.6

data = [1 2.3 200 4; 2 2 100 5.6]
X = make_sparse(data)

@test size(X,1) == 2
@test size(X,2) == 200
@test length(X.nzval) == 4
@test X[2,100] == 5.6