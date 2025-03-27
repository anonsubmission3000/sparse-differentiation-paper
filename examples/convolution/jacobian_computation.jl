using Pkg
Pkg.activate(@__DIR__)

using SparseConnectivityTracer  # sparsity detection
using SparseMatrixColorings     # sparsity pattern coloring
using DifferentiationInterface  # common interface to AD backends
using ForwardDiff               # forward-mode AD backend
using Flux                      # deep learning framework

# Specify global sparsity detection and coloring algorithm
detector = TracerSparsityDetector()
coloring = GreedyColoringAlgorithm()

# Specify AD and ASD backends
ad_backend = AutoForwardDiff()
asd_backend = AutoSparse(
    AutoForwardDiff(); sparsity_detector=detector, coloring_algorithm=coloring
)

# Create input tensor and convolutional layer
x = randn(Float32, 10, 10, 3, 1)
layer = Conv((5, 5), 3 => 1)

# Compute Jacobian
jacobian(layer, ad_backend, x)   # using AD
jacobian(layer, asd_backend, x)  # using ASD
