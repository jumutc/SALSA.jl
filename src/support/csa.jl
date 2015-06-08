function csa(obj_fun, pn)
    #
    # Internal function based on 
    # Xavier-de-Souza S, Suykens JA, Vandewalle J, Bolle D., Coupled Simulated
    # Annealing, IEEE Trans Syst Man Cybern B Cybern. 2010 Apr;40(2):320-35.
    #
    # Copyright (c) 2014,  KULeuven-ESAT-SCD, License & help @% http://www.esat.kuleuven.be/sista/SALSA

    T0 = 1
    Tac0 = 1
    FEmax = 40
    FTsteps = 20
    etol = 1e-45

    #clear OPT
    pdim = size(pn,1);
    pnum = size(pn,2);

    NT = ceil(FEmax/FTsteps/pnum);  # max. number of cooling cycles
    NI = FTsteps;  #steps per temperature

    #srand(hash(sum(pn)*time()))

    e0 = float(obj_fun(pn))

    p0 = pn;
    be0 = minimum(e0);
    ind = indmin(e0)
    bp0 = pn[:,ind];

    pblty = zeros(1,pnum);
    sgnCR = -1;
    CR = 0.1;#0.05;
    pvar_est = 0.995;

    Tac = Tac0;

    for k = 1:NT
        pbltvar = var(pblty);
        sgnCR_ant = sgnCR;
        sgnCR = 2*((pbltvar > (pvar_est*(pnum-1)/(pnum^2)))-0.5);
        Tac = Tac + sgnCR*CR*Tac;

        # T schedules
        T = T0/k;
    
        for l = 1:NI
            # choose new coordinates and compute
            # the function to minimize
            r = tan(pi*(rand(pdim,pnum).-0.5));
            # ****************** Wrapping ***************
            #pn = 2*mod((p0 + r * T + 1)/2,1)-1;
            #**************** Non wrapping *************
             pn = p0 + r * T ;#* (diag(e0)./sum(e0));
             indd = find(abs(pn).>10);
             while length(indd) > 0
                 r[indd] = tan(pi*(rand(size(indd)).-0.5));
                 pn = p0 + r * T ;#* (diag(e0)./sum(e0));
                 indd = find(abs(pn).>15);
             end

            en = float(obj_fun(pn))

            Esum = sum(exp((e0.-maximum(e0))./Tac));
            for i=1:pnum
                pblty[i] = min(1.0,exp((e0[i]-maximum(e0))/Tac)/Esum);
                if pblty[i] > 1
                    pblty[i] = 1.
                end
                
                if ( en[i] - e0[i] ) < 0
                    # accept
                    p0[:,i] = pn[:,i]; 
                    e0[i] = en[i];
                    if e0[i] < be0
                        be0 = e0[i]; 
                        bp0 = p0[:,i]; 
                   end
                else
                    r = rand();
                    if pblty[i] >= r
                        # accept
                        p0[:,i] = pn[:,i]; 
                        e0[i] = en[i];
                    end
                end
            end
            if any(e0.<etol)
                break
            end
        end
        if any(e0.<etol)
            break
        end
    end

    efinal = be0;
    pfinal = bp0;
    efinal, pfinal
end
