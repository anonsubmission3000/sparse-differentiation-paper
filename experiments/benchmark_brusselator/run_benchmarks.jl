using DrWatson # YES
quickactivate(@__DIR__)

using ADTypes
using SparseConnectivityTracer
using Symbolics
using DifferentiationInterface
using ForwardDiff: ForwardDiff
using LinearMaps, IterativeSolvers

# Save sparsity pattern as image
using SparseMatrixColorings
using SparseMatrixColorings: show_colors
using Colors, ColorSchemes
using ImageIO

using DataFrames
using Chairmarks

## Brusselator definition
include("brusselator.jl")

const detector_sct = TracerSparsityDetector()
const detector_sym = SymbolicsSparsityDetector()

const coloring_algorithm = GreedyColoringAlgorithm(; decompression=:direct)

const backend_ad = AutoForwardDiff()
const backend_asd_sct = AutoSparse(
    backend_ad; sparsity_detector=detector_sct, coloring_algorithm=coloring_algorithm
)
const backend_asd_sym = AutoSparse(
    backend_ad; sparsity_detector=detector_sym, coloring_algorithm=coloring_algorithm
)

const cm_seconds = 10 # minimum amount of time in seconds to run Chairmarks benchmarks 

#======================#
# Benchmark definition #
#======================#

function run_brusselator_benchmarks(d::Dict)
    @unpack N = d

    # Prepare benchmark
    f! = Brusselator!(N)
    x = copy(f!.u0)
    y = similar(x)
    f!(y, x)
    rhs = copy(y)

    # Prepare backends
    prep_ad_pushforward = prepare_pushforward(f!, y, backend_ad, x, (zero(x),))
    prep_ad = prepare_jacobian(f!, y, backend_ad, x)
    prep_asd_sct = prepare_jacobian(f!, y, backend_asd_sct, x)
    prep_asd_sym = prepare_jacobian(f!, y, backend_asd_sym, x)

    pushforward!_linear_map = LinearMap(
        (res, v) -> pushforward!(f!, y, (res,), prep_ad_pushforward, backend_ad, x, (v,)),
        length(y),
        length(x);
        ismutating=true,
        issymmetric=false,
    )

    @info "Computing initial sparsity pattern..."
    pattern = sparsity_pattern(prep_asd_sct)

    # Compute metadata
    rows, cols = size(pattern)
    entries = rows * cols
    n_zeros = sum(iszero, pattern)
    rel_sparsity = n_zeros / entries
    n_colors_col = maximum(column_colors(prep_asd_sct))
    @info "Pattern metadata:" rows cols n_zeros rel_sparsity n_colors_col

    # Dry-run to ensure Jacobians match
    @info "Checking output correctness..."
    J_ad = jacobian(f!, y, prep_ad, backend_ad, x)
    J_asd_sct = jacobian(f!, y, prep_asd_sct, backend_asd_sct, x)
    J_asd_sym = jacobian(f!, y, prep_asd_sym, backend_asd_sym, x)
    if !isapprox(J_ad, J_asd_sct)
        error("AD and AST (SCT) Jacobians don't match.")
    end
    if !isapprox(J_ad, J_asd_sym)
        error("AD and AST (Symbolics) Jacobians don't match.")
    end

    # Dry run to ensure linear solves match
    res_op_iter = bicgstabl(pushforward!_linear_map, rhs)
    res_sparse_iter = bicgstabl(J_asd_sct, rhs)
    res_sparse_direct = J_asd_sct \ rhs
    @assert isapprox(res_op_iter, res_sparse_iter; rtol=1e-5)
    @assert isapprox(res_sparse_direct, res_sparse_iter; rtol=1e-5)

    ## Color Matrix to save image
    if N <= 24
        @info "Saving image of colored matrix..."
        pattern = jacobian_sparsity(f!, y, x, detector_sct)
        problem = ColoringProblem(; structure=:nonsymmetric, partition=:column)
        result = coloring(pattern, problem, coloring_algorithm)
        ncolors = 10 # max colors

        colorscheme = get(ColorSchemes.rainbow, range(0.0, 1.0; length=ncolors))
        img, _ = show_colors(result; colorscheme=colorscheme, scale=5, pad=1)
        path = plotsdir("brusselator_coloring", "N=$(N).png")
        save(path, img)
        @info "... saved rainbow-colored image to $path."

        colorscheme = distinguishable_colors(ncolors, [RGB(1, 1, 1)]; dropseed=true)
        img, _ = show_colors(result; colorscheme=colorscheme, scale=5, pad=1)
        path = plotsdir("brusselator_coloring", "N=$(N)_distinguishable.png")
        save(path, img)
        @info "... saved distinguishable colors image to $path."

        colorscheme = [RGBA(0, 0, 0, 0)]
        img, _ = show_colors(result; colorscheme=colorscheme, scale=5, pad=1, warn=false)
        path = plotsdir("brusselator_coloring", "N=$(N)_bw.png")
        save(path, img)
        @info "... saved black & white image to $path."
    end

    ## Run benchmarks
    @info "Running sparsity detection benchmarks..."
    bm_sp_sct = @b jacobian_sparsity($f!, $y, $x, $detector_sct) seconds = cm_seconds
    bm_sp_sym = @b jacobian_sparsity($f!, $y, $x, $detector_sym) seconds = cm_seconds

    @info "Running AD benchmarks..."

    bm_ad_preparation = @b prepare_jacobian($f!, $y, $backend_ad, $x) seconds = cm_seconds
    @info "...AD preparation:" bm_ad_preparation.time

    bm_ad_execution = @b jacobian($f!, $y, $prep_ad, $backend_ad, $x) seconds = cm_seconds
    @info "...AD execution:" bm_ad_execution.time

    bm_asd_sct_preparation = @b prepare_jacobian($f!, $y, $backend_asd_sct, $x) seconds =
        cm_seconds
    @info "...ASD (SCT) preparation:" bm_asd_sct_preparation.time

    bm_asd_sct_execution = @b jacobian($f!, $y, $prep_asd_sct, $backend_asd_sct, $x) seconds =
        cm_seconds
    @info "...ASD (SCT) execution:" bm_asd_sct_execution.time

    bm_asd_sym_preparation = @b prepare_jacobian($f!, $y, $backend_asd_sym, $x) seconds =
        cm_seconds
    @info "...ASD (Symbolics) preparation:" bm_asd_sym_preparation.time

    bm_asd_sym_execution = @b jacobian($f!, $y, $prep_asd_sym, $backend_asd_sym, $x) seconds =
        cm_seconds
    @info "...ASD (Symbolics) execution:" bm_asd_sym_execution.time

    bm_linsolve_operator_iterative = @b bicgstabl($pushforward!_linear_map, $rhs)
    @info "...Linsolve (lazy operator, iterative solver)" bm_linsolve_operator_iterative.time

    bm_linsolve_sparse_iterative = @b bicgstabl($J_asd_sct, $rhs)
    @info "...Linsolve (sparse matrix, iterative solver)" bm_linsolve_sparse_iterative.time

    bm_linsolve_sparse_direct = @b \($J_asd_sct, $rhs)
    @info "...Linsolve (sparse matrix, direct solver)" bm_linsolve_sparse_direct.time

    ## Save benchmark results
    res = Dict(
        :N => N,
        :time_ad_prep => bm_ad_execution.time,
        :time_ad_noprep => bm_ad_preparation.time + bm_ad_execution.time,
        :time_asd_sct_prep => bm_asd_sct_execution.time,
        :time_asd_sct_noprep => bm_asd_sct_preparation.time + bm_asd_sct_execution.time,
        :time_asd_sym_prep => bm_asd_sym_execution.time,
        :time_asd_sym_noprep => bm_asd_sym_preparation.time + bm_asd_sym_execution.time,
        :time_sp_sct => bm_sp_sct.time,
        :time_sp_sym => bm_sp_sym.time,
        :time_ls_op_iter => bm_linsolve_operator_iterative.time,
        :time_ls_sp_iter => bm_asd_sct_execution.time + bm_linsolve_sparse_iterative.time,
        :time_ls_sp_direct => bm_asd_sct_execution.time + bm_linsolve_sparse_direct.time,
        :rows => rows,
        :cols => cols,
        :n_zeros => n_zeros,
        :rel_sparsity => rel_sparsity,
        :n_colors_col => n_colors_col,
    )
    return res
end

#====================#
# Run all benchmarks #
#====================#

allparams = Dict(:N => 3 * 2 .^ (1:6))
experiments = dict_list(allparams)
nexperiments = length(experiments)

for (i, d) in enumerate(experiments)
    @info "Running experiment $i/$nexperiments:" d

    # Run experiment if it doesn't exist yet, save result to file with git commit info
    res, file = produce_or_load(run_brusselator_benchmarks, d, datadir("results"); tag=true)
    yield()
end
