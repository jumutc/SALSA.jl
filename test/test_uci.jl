using SALSA, NHST, Distributions

datasets = ["pendigits.csv", "optdigits.csv", "shuttle.txt", "spambase.txt", "magic04.csv"]
delims = [',', ',', ' ', ' ', ',']
path = "~"

f_gen = open(string(path,"gen.csv"),"a+")
f_spar = oopen(string(path,"spar.csv","a+")
f_gen_dump = open(string(path,"gen_dump.csv","a+")
f_spar_dump = open(string(path,"spar_dump.csv","a+")
f_pval = open(string(path,"pval.csv","a+")

writecsv(f_gen, ["dataset" "(r)l1-RDA" "std" "(r)l2-RDA" "std" "l1-RDA" "std" "Pegasos" "std" "Drop-out" "std"])
writecsv(f_spar, ["dataset" "(r)l1-RDA" "std" "(r)l2-RDA" "std" "l1-RDA" "std" "Pegasos" "std" "Drop-out" "std"])

for (d, ds) in zip(delims, datasets)
 
    X_tmp = readdlm(string("/users/stadius/vjumutc/Research/Data2/",ds),d,Float64)
    Xlbl = X_tmp[:,end]
    Xtrn = X_tmp[:,1:end-1]
    Xlbl[Xlbl.!=1] = -1

    n_folds = 50
    (N,d) = size(Xtrn)
    tenth = int(N/10)
    pd = Categorical(N)
    space = 1:1:N

    test_error_rl1rda = zeros(n_folds)
    test_error_rl2rda = zeros(n_folds)     
    test_error_l1rda = zeros(n_folds)
    test_error_peg = zeros(n_folds)
    test_error_dout = zeros(n_folds)  

    sparsity_rl1rda = zeros(n_folds)
    sparsity_rl2rda = zeros(n_folds)
    sparsity_l1rda = zeros(n_folds)
    sparsity_peg = zeros(n_folds)
    sparsity_dout = zeros(n_folds)

        
    for x=1:n_folds
        @printf "\nFold number = %d\n" x
        
        test_set = Set(rand(pd,tenth))
        # the most effective way to find and treat indices
        test  = [i ∈ test_set ? true : false for i in space]

        Xtest = Xtrn[test,:]
        X     = Xtrn[~test,:]  
        Ytest = Xlbl[test]
        Y     = Xlbl[~test]

        # Reweighted l1-RDA PART
        @time model = salsa(R_L1RDA,X,Y,Xtest)
        error = 1-mean(Ytest .== model.Ytest)
        sparsity_t = mean(model.w .!= 0)
        
        @printf "\nReweighted l1-RDA test error = %.3f\n" error 
        @printf "\nReweighted l1-RDA test sparsity = %.3f\n" sparsity_t 
        
        test_error_rl1rda[x] = error
        sparsity_rl1rda[x] = sparsity_t 

        # Reweighted l2-RDA PART
        @time model = salsa(R_L2RDA,X,Y,Xtest)
        error = 1-mean(Ytest .== model.Ytest)
        sparsity_t = mean(model.w .!= 0)
        
        @printf "\nReweighted l2-RDA test error = %.3f\n" error 
        @printf "\nReweighted l2-RDA test sparsity = %.3f\n" sparsity_t 
        
        test_error_rl2rda[x] = error
        sparsity_rl2rda[x] = sparsity_t 
        
        # l1-RDA PART
        @time model = salsa(L1RDA,X,Y,Xtest)
        error = 1-mean(Ytest .== model.Ytest)
        sparsity_t = mean(model.w .!= 0)
        
        @printf "\nl1-RDA test error = %.3f\n" error 
        @printf "\nl1-RDA test sparsity = %.3f\n" sparsity_t 
        
        test_error_l1rda[x] = error
        sparsity_l1rda[x] = sparsity_t 

        #Drop-off PART
        @time model = salsa(DROP_OUT,X,Y,Xtest)
        error = 1-mean(Ytest .== model.Ytest)
        sparsity_t = mean(model.w .!= 0)
        
        @printf "\nDrop-off test error = %.3f\n" error 
        @printf "\nDrop-off test sparsity = %.3f\n" sparsity_t 
        test_error_dout[x] = error
        sparsity_dout[x] = sparsity_t 

        #Pegasos PART
        @time model = salsa(X,Y,Xtest)
        error = 1-mean(Ytest .== model.Ytest)
        sparsity_t = mean(model.w .!= 0)
        
        @printf "\nPegasos test error = %.3f\n" error 
        @printf "\nPegasos test sparsity = %.3f\n" sparsity_t 
        
        test_error_peg[x] = error
        sparsity_peg[x] = sparsity_t 
    end


    @printf "\n====================Results for dataset: %s============================" ds

    @printf "\nReweighted l1-RDA avg test error: %.5f stddev: %.5f" mean(test_error_rl1rda) std(test_error_rl1rda)
    @printf "\nReweighted l2-RDA avg test error: %.5f stddev: %.5f" mean(test_error_rl2rda) std(test_error_rl2rda)
    @printf "\nPegasos (ref. impl.) avg test error: %.5f stddev: %.5f" mean(test_error_peg) std(test_error_peg)
    @printf "\nPegasos (with drop-off) avg test error: %.5f stddev: %.5f" mean(test_error_dout) std(test_error_dout)
    @printf "\nl1-RDA avg test error: %.5f stddev: %.5f" mean(test_error_l1rda) std(test_error_l1rda)

    @printf "\nReweighted l1-RDA avg test sparsity: %.5f stddev: %.5f" mean(sparsity_rl1rda) std(sparsity_rl1rda)
    @printf "\nReweighted l2-RDA avg test sparsity: %.5f stddev: %.5f" mean(sparsity_rl2rda) std(sparsity_rl2rda)
    @printf "\nPegasos (ref. impl.) avg test sparsity: %.5f stddev: %.5f" mean(sparsity_peg) std(sparsity_peg)
    @printf "\nPegasos (with drop-off) avg test sparsity: %.5f stddev: %.5f" mean(sparsity_dout) std(sparsity_dout)
    @printf "\nl1-RDA avg test sparsity: %.5f sparsity: %.5f" mean(sparsity_l1rda) std(sparsity_l1rda)


    p1 = run(TTest, test_error_l1rda, test_error_rl1rda).p_value
    p2 = run(TTest, test_error_l1rda, test_error_rl2rda).p_value
    p3 = run(TTest, test_error_rl1rda, test_error_rl2rda).p_value
    p4 = run(TTest, test_error_peg, test_error_dout).p_value

    p5 = run(TTest, sparsity_l1rda, sparsity_rl1rda).p_value
    p6 = run(TTest, sparsity_l1rda, sparsity_rl2rda).p_value
    p7 = run(TTest, sparsity_rl1rda, sparsity_rl2rda).p_value


    @printf "\nReweighted l1-RDA vs. l1-RDA test error ttest2 p-value: %.5f" p1
    @printf "\nReweighted l2-RDA vs. l1-RDA test error ttest2 p-value: %.5f" p2
    @printf "\nReweighted l2-RDA vs. Reweighted l1-RDA test error ttest2 p-value: %.5f" p3
    @printf "\nPegasos (ref. impl.) vs. Pegasos (with drop-off) test error ttest2 p-value: %.5f" p4

    @printf "\nReweighted l1-RDA vs. l1-RDA test sparsity ttest2 p-value: %.5f" p5
    @printf "\nReweighted l2-RDA vs. l1-RDA test sparsity ttest2 p-value: %.5f" p6
    @printf "\nReweighted l2-RDA vs. Reweighted l1-RDA test sparsity ttest2 p-value: %.5f" p7

    writecsv(f_gen, [ds mean(test_error_rl1rda) std(test_error_rl1rda) mean(test_error_rl2rda) std(test_error_rl2rda) mean(test_error_l1rda) std(test_error_l1rda) mean(test_error_peg) std(test_error_peg) mean(test_error_dout) std(test_error_dout)])
    writecsv(f_spar, [ds mean(sparsity_rl1rda) std(sparsity_rl1rda) mean(sparsity_rl2rda) std(sparsity_rl2rda) mean(sparsity_l1rda) std(sparsity_l1rda) mean(sparsity_peg) std(sparsity_peg) mean(sparsity_dout) std(sparsity_dout)])
    writecsv(f_pval, [ds p1 p2 p3 p4 p5 p6 p7])

    writecsv(f_gen_dump, [ds "Reweighted l1-RDA"  test_error_rl1rda'
                          ds "Reweighted l2-RDA"  test_error_rl2rda'
                          ds "l1-RDA"             test_error_l1rda'
                          ds "Pegasos (drop-off)" test_error_dout'
                          ds "Pegasos"            test_error_peg'])

    writecsv(f_spar_dump, [ds "Reweighted l1-RDA"  sparsity_rl1rda'
                           ds "Reweighted l2-RDA"  sparsity_rl2rda'
                           ds "l1-RDA"             sparsity_l1rda'
                           ds "Pegasos (drop-off)" sparsity_dout'
                           ds "Pegasos"            sparsity_peg'])

    flush(f_gen); flush(f_spar); flush(f_pval); flush(f_gen_dump); flush(f_spar_dump)
end

close(f_gen)
close(f_spar)
close(f_pval)
close(f_gen_dump)
close(f_spar_dump)