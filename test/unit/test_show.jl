using SALSA, Base.Test

outOriginal = STDOUT
(outRead, outWrite) = redirect_stdout()
model = SALSAModel(LINEAR,PEGASOS(),HINGE)
model.output.dfunc = loss_derivative(HINGE)
show(outWrite, model)

s = utf8(readavailable(outRead))
redirect_stdout(outOriginal)

@test contains(s,"SALSA model:")
@test contains(s,"SALSA model.output:")
@test contains(s,"PEGASOS")
@test contains(s,"LINEAR")
@test contains(s,"HINGE")
@test contains(s,"dfunc")