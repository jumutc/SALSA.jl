function entropysubset{K <: Kernel}(X, k::K, represent_points)
    # Performs Renyi Entropy based representative points selection

    capacity = ceil(represent_points)
    selectionset = randperm(size(X,1))
    sv = selectionset[1:capacity]
    svX = X[sv,:]

    # Calculating the Renyi for pre-determined points
    totalinfo = zeros(convert(Int,capacity),2)
    totalinfo[:,2] = sum(kernel_matrix(k,svX),2)
    totalinfo[:,1] = 1:1:capacity

    capsquare = capacity^2
    totalcrit = sum(totalinfo[:,2])
    logtotalcrit = -log(totalcrit/capsquare)

    max_c = logtotalcrit
    # Maximixing the quadratic Renyi Entropy
    for i=1:size(X,1)
        # Find the smallest entropy
        val, id = findmin(totalinfo[:,2])
        # Subtract from totalcrit once for row and once for column 
        # and add 1 for diagonal term which is subtracted twice
        temptotalcrit = totalcrit - 2*val + 1
        #Try to evaluate kernel function 
        if (id == capacity) 
            sub = totalinfo[1:id-1,1]
        elseif (id == 1)
            sub = totalinfo[id+1:end,1]
        else   
            sub = totalinfo[[1:id-1;id+1:end],1]
        end
        distance_eval = sum(kernel_matrix(k,X[i,:],svX[sub,:]))
        # Add to totalcrit once for row and once for column 
        # and subtract 1 for diagonal term which is added twice
        temptotalcrit = temptotalcrit + 2*distance_eval - 1
        logtemptotalcrit = -log(temptotalcrit/capsquare)
        # Evaluate that Renyi Entropy has increased 
        if (max_c <= logtemptotalcrit)
            totalinfo[id,2] = distance_eval
            totalcrit = temptotalcrit
            max_c = logtemptotalcrit
            sv[id] = i
        end
    end
   
    sv
end
