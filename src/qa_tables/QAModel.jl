abstract QANode

type LinearQANode <: QANode
	question:: UTF32String
	procfunc:: Function
	# field not provided by default at instantiation
	options:: Dict
	answer:: ASCIIString
end

immutable QAOption{N <: QANode}
	updatefunc:: Function
	next:: Nullable{N}
end

LinearQANode(question, procfunc) = LinearQANode(question,procfunc,Dict(),"") 