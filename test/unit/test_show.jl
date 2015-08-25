using SALSA, Base.Test

outOriginal = STDOUT
(outRead, outWrite) = redirect_stdout()
model = SALSAModel(LINEAR,PEGASOS(),HINGE)
show(outWrite, model)

s = char(readavailable(outRead))
redirect_stdout(outOriginal)
println(s)

@test contains(s,"SALSA model:")
@test contains(s,"SALSA model.output:")
@test contains(s,"PEGASOS")
@test contains(s,"LINEAR")
@test contains(s,"HINGE")