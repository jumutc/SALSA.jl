using SALSA, Distances, Base.Test

outOriginal = STDOUT
(outRead, outWrite) = redirect_stdout()

read_char = () -> '\n'
read_int = () -> '\n'
run_model = (X,Y,model,Xtest) -> begin
	# test output model 
	@test model.mode == LINEAR
	@test model.algorithm == PEGASOS()
	@test model.global_opt == CSA()
	@test model.loss_function == HINGE
	@test model.validation_criterion == MISCLASS()
end

show(salsa_qa(eye(10),read_char,read_int,run_model))
s = String(readavailable(outRead))

#test Q/A table contents
@test contains(s, "Computing the model..")
@test contains(s, "(or ENTER for default)")

read_char = () -> 'n'
read_int = () -> 1
run_model = (X,Y,model,Xtest) -> begin
	# test output model 
	@test model.mode == LINEAR
	@test typeof(model.algorithm) <: RK_MEANS
	@test model.algorithm.support_alg == DROP_OUT
	@test model.algorithm.metric == Euclidean()
	@test model.algorithm.k_clusters == 1
	@test model.algorithm.max_iter == 1
	@test model.global_opt == CSA()
	@test model.loss_function == LEAST_SQUARES
	@test model.validation_criterion == SILHOUETTE()
	@test model.max_cv_iter == 1
	@test model.max_iter == 1
end

salsa_qa(eye(10),read_char,read_int,run_model)
redirect_stdout(outOriginal)