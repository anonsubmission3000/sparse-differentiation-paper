using Pkg
Pkg.activate(@__DIR__)

using SparseConnectivityTracer, Flux   # import required packages

x = randn(Float32, 10, 10, 3, 1)       # create input tensor
layer = Conv((5, 5), 3 => 1)           # create convolutional layer

detector = TracerSparsityDetector()    # specify global sparsity pattern
jacobian_sparsity(layer, x, detector)  # compute pattern
