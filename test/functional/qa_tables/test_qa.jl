using SALSA, Base.Test

outOriginal = STDOUT
(outRead, outWrite) = redirect_stdout()

read_char = () -> '\n'
read_int = () -> '\n'

Xf = readcsv(joinpath(Pkg.dir("SALSA"),"data","iris.data.csv"))
show(salsa_qa(Xf,read_char,read_int))

s = utf8(readavailable(outRead))
redirect_stdout(outOriginal)

#test Q/A table
@test contains(s, "Computing the model..")
@test contains(s, "(or ENTER for default)")
# test output model 
@test contains(s,"SALSA model:")
@test contains(s,"SALSA model.output:")
@test contains(s,"HINGE")
@test contains(s,"PEGASOS")
@test contains(s,"LINEAR")
@test contains(s,"dfunc")