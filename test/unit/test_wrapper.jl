using SALSA, Base.Test

X = DelimitedFile(joinpath(Pkg.dir("SALSA"),"data","iris.data.csv"),false,',')
X_ref = readcsv(joinpath(Pkg.dir("SALSA"),"data","iris.data.csv"))

@test size(X) == (150,5)
@test size(X,1) == 150
@test size(X,2) == 5

@test X[11,3] == X_ref[11,3]
@test getindex(X,(11,3)) == X_ref[11,3]
@test sub(X,[1,3,5],:) == sub(X_ref,[1,3,5],:)