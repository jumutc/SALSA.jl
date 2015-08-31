using SALSA, Base.Test, Base.convert

outOriginal = STDOUT
(outRead, outWrite) = redirect_stdout()
model = SALSAModel(LINEAR,PEGASOS(),HINGE)
model.output.dfunc = loss_derivative(HINGE)
show(outWrite, model)

s = readavailable(outRead)
redirect_stdout(outOriginal)

if typeof(s) != ASCIIString
	s = UTF8String(s)
end

@test contains(s,"SALSA model:")
@test contains(s,"SALSA model.output:")
@test contains(s,"PEGASOS")
@test contains(s,"LINEAR")
@test contains(s,"HINGE")
@test contains(s,"dfunc")