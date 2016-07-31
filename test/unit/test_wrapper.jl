using SALSA, Base.Test

X = DelimitedFile(joinpath(dirname(@__FILE__),"..","..","data","iris.data.csv"),false,',')
X_ref = readcsv(joinpath(dirname(@__FILE__),"..","..","data","iris.data.csv"))

@test size(X) == (150,5)
@test size(X,1) == 150
@test size(X,2) == 5

X = DelimitedFile(joinpath(dirname(@__FILE__),"..","..","data","iris.data.csv"),false,',')
@test getindex(X,(1,3)) == X_ref[1,3]

# we preserve only row-by-row readthrough of files (for now)
X = DelimitedFile(joinpath(dirname(@__FILE__),"..","..","data","iris.data.csv"),false,',')
@test sub(X,[1,3,5],:) == sub(X_ref,[1,2,3],:)