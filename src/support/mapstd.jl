function mapstd(datain::Matrix{Float64})
	# apply mapstd
	means = mean(datain,1)
	stds  = std(datain,1)
	stds[stds.==0] = 1

	dataout = datain - ones(size(datain,1),1)*means
	dataout = dataout./(ones(size(datain,1),1)*stds)
	(dataout, means, stds)
end

function mapstd(datain::SparseMatrixCSC)
	(datain, 0, 0)
end

function mapstd(datain::SparseMatrixCSC,mean,std)
	datain
end

function mapstd(datain::Matrix{Float64},mean,std)
	# apply mapstd
	dataout = datain - ones(size(datain,1),1)*mean
	dataout = dataout./(ones(size(datain,1),1)*std)
	dataout
end