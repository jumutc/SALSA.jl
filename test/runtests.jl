tests = ["unit/test_pegasos", 
		 "unit/test_wrapper", 
		 "functional/test_linear",
		 "functional/test_nonlinear",
		 "functional/test_wrapper"]

print_with_color(:blue, "Running tests:\n")

for t in tests
	test_fn = "$t.jl"
	print_with_color(:green, "* $test_fn\n")
	include(test_fn)
end