function salsa_qa{N <: Number}(X::Matrix{N})
	proto = [SALSAModel();[]]
	read_char = () -> lowercase(readline(STDIN)[1])
	read_int  = () -> parse(Int,readline(STDIN)[1])
	get_key   = (ans) -> typeof(ans) == Char ? ans : 'y'

	n1 = LinearQANode("\nDo you have any target variable of interest in X? [y/n]: ", read_char)
	n2 = LinearQANode("\nPlease provide the column number of your target variable: ", read_int)
	n3 = LinearQANode("\nIs your problem of the classification type? [y/n]: ", read_char)
	n4 = LinearQANode("\nClustering mode is selected. Please provide the number of clusters: ", read_int)
	n5 = LinearQANode("\nPlease select a loss function from options\n $(print_opts(loss_opts))\n: ", read_int)
	n6 = LinearQANode("\nDo you want to perform Nyström (nonlinear) approximation? [y/n]: ", read_char)
	n7 = LinearQANode("\nPlease select an algorithm from options\n $(print_opts(algo_opts))\n: ", read_int)
	n8 = LinearQANode("\nPlease select a cross-validation (CV) criterion from options\n $(print_opts(criteria_opts))\n: ", read_int)
	n9 = LinearQANode("\nPlease select a global optimization method from options\n $(print_opts(optim_opts))\n: ", read_int)
	n10= LinearQANode("\nPlease select a type of Kernel for Nyström approximation from options\n $(print_opts(kernel_opts))\n: ", read_int)

	n1.options = Dict('y' => QAOption((ans) -> Void, Nullable(n2)),
					  'n' => QAOption((ans) -> Void, Nullable(n4)))
	n2.options = Dict('y' => QAOption((ans) -> append!(proto,[setdiff(1:size(X,2),ans);ans]), Nullable(n3)))
	n3.options = Dict('y' => QAOption((ans) -> Void, Nullable(n5)),
					  'n' => QAOption((ans) -> begin proto[1] = SALSAModel(LEAST_SQUARES,proto[1]);
					  						   proto[1].process_labels = false end , Nullable(n6)))
	n4.options = Dict('y' => QAOption((ans) -> proto[1] = SALSAModel(LINEAR,RK_MEANS(PEGASOS,ans,20,Euclidean()),LEAST_SQUARES,
											   validation_criteria=SILHOUETTE(),global_opt=DS([1]),process_labels=false), Nullable()))
	n5.options = Dict('y' => QAOption((ans) -> proto[1] = SALSAModel(loss_opts[ans],proto[1]), Nullable(n6)))
	n6.options = Dict('n' => QAOption((ans) -> proto[1] = SALSAModel(mode_opts[ans],proto[1]), Nullable(n7)),
					  'y' => QAOption((ans) -> proto[1] = SALSAModel(mode_opts[ans],proto[1]), Nullable(n10)))
	n7.options = Dict('y' => QAOption((ans) -> proto[1] = SALSAModel(algo_opts[ans],proto[1]), Nullable(n8)))
	n8.options = Dict('y' => QAOption((ans) -> proto[1].validation_criteria = criteria_opts[ans], Nullable(n9)))
	n9.options = Dict('y' => QAOption((ans) -> proto[1].global_opt = optim_opts[ans], Nullable()))
	n10.options= Dict('y' => QAOption((ans) -> proto[1] = SALSAModel(kernel_opts[ans],proto[1]), Nullable(n7)))

	current = Nullable(n1)

	while ~isnull(current)
		node = get(current)
		print_with_color(:blue,node.question)
		ans = node.procfunc()
		node.answer = "$ans"
		key = get_key(ans)
		option = node.options[key]
		option.updatefunc(ans)
		current = option.next
	end

	if length(proto) == 1
		Y = ones(size(X,1),1)
	else
		Y = X[:,convert(Int,proto[end])]
		X = X[:,convert(Array{Int},proto[2:end-1])]
	end

	@printf "\nComputing the model...\n"
	salsa(X,Y,proto[1],X)
end

function print_opts(opts)
	s = ""
	for (key, val) in opts
		s = string(s,"\t$key : $val\n")
	end
	s
end