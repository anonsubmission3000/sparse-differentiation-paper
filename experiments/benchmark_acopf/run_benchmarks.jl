#= Read README before running! =#
using DrWatson
quickactivate(@__DIR__)

using ADTypes
using Symbolics
using SparseConnectivityTracer
using SparseConnectivityTracer: HessianTracer, DictHessianPattern, NotShared
using SparseMatrixColorings
using DifferentiationInterface
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff

using Chairmarks

# Load OPF cases
include("load_opf.jl")

const HT = HessianTracer{DictHessianPattern{Int,Set{Int},Dict{Int,Set{Int}},NotShared}}
const detector_sct = TracerSparsityDetector(; hessian_tracer_type=HT)
const detector_sym = SymbolicsSparsityDetector()
const coloring_algorithm = GreedyColoringAlgorithm(; decompression=:direct)

const backend_dense = SecondOrder(AutoForwardDiff(), AutoReverseDiff())
const backend_sparse = AutoSparse(
    backend_dense; sparsity_detector=detector_sct, coloring_algorithm=coloring_algorithm
)

const cm_seconds = 10 # minimum amount of time to run Chairmarks benchmarks 

#======================#
# Benchmark definition #
#======================#

function strip_file_name(path)
    file = split(path, '/')[end]
    return chopsuffix(file, ".m")
end

function run_opf_benchmarks(d::Dict)
    @unpack case_file = d
    lagrangian, x, n_vars, n_cons = load_opf_case(case_file)

    # Prepare backends
    prep_dense = prepare_hessian(lagrangian, backend_dense, x)
    prep_sparse = prepare_hessian(lagrangian, backend_sparse, x)

    @info "Computing initial sparsity pattern..."
    pattern = sparsity_pattern(prep_sparse)

    # Compute metadata
    rows, cols = size(pattern)
    entries = rows * cols
    n_zeros = sum(iszero, pattern)
    rel_sparsity = n_zeros / entries
    n_colors_col = maximum(column_colors(prep_sparse))
    @info "Pattern metadata:" rows cols n_zeros rel_sparsity n_colors_col

    # Dry-run to ensure Hessians match
    @info "Checking output correctness..."
    H_dense = hessian(lagrangian, prep_dense, backend_dense, x)
    H_sparse = hessian(lagrangian, prep_sparse, backend_sparse, x)
    if !isapprox(H_dense, H_sparse)
        error("Hessians don't match.")
    end

    @info "Running sparsity detection benchmarks..."
    bm_sp_sct = @b hessian_sparsity($lagrangian, $x, $detector_sct) seconds = cm_seconds
    @info "...Detection SCT:" bm_sp_sct.time
    bm_sp_sym = @b hessian_sparsity($lagrangian, $x, $detector_sym) seconds = cm_seconds
    @info "...Detection Symbolics:" bm_sp_sym.time

    @info "Running AD benchmarks..."
    bm_dense_preparation = @b prepare_hessian($lagrangian, $backend_dense, $x) seconds =
        cm_seconds
    @info "...Dense AD (preparation):" bm_dense_preparation.time
    bm_dense_execution = @b hessian($lagrangian, $prep_dense, $backend_dense, $x) seconds =
        cm_seconds
    @info "...Dense AD (execution):" bm_dense_execution.time
    bm_sparse_preparation = @b prepare_hessian($lagrangian, $backend_sparse, $x) seconds =
        cm_seconds
    @info "...Sparse AD (preparation):" bm_sparse_preparation.time
    bm_sparse_execution = @b hessian($lagrangian, $prep_sparse, $backend_sparse, $x) seconds =
        cm_seconds
    @info "...Sparse AD (execution):" bm_sparse_execution.time

    # Save benchmark results
    res = Dict(
        :case_name => strip_file_name(case_file),
        :time_dense_prepped => bm_dense_execution.time,
        :time_sparse_prepped => bm_sparse_execution.time,
        :time_sparse_unprepped => bm_sparse_preparation.time + bm_sparse_execution.time,
        :time_sp_sct => bm_sp_sct.time,
        :time_sp_sym => bm_sp_sym.time,
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

case_files = [
    "pglib_opf_case3_lmbd.m",
    "pglib_opf_case5_pjm.m",
    "pglib_opf_case14_ieee.m",
    "pglib_opf_case24_ieee_rts.m",
    "pglib_opf_case30_as.m",
    "pglib_opf_case30_ieee.m",
    "pglib_opf_case39_epri.m",
    "pglib_opf_case57_ieee.m",
    "pglib_opf_case60_c.m",
    "pglib_opf_case73_ieee_rts.m",
    "pglib_opf_case89_pegase.m",
    "pglib_opf_case118_ieee.m",
    "pglib_opf_case162_ieee_dtc.m",
    "pglib_opf_case179_goc.m",
    "pglib_opf_case197_snem.m",
    "pglib_opf_case200_activ.m",
    "pglib_opf_case240_pserc.m",
    "pglib_opf_case300_ieee.m",
    "pglib_opf_case500_goc.m",
    "pglib_opf_case588_sdet.m",
    "pglib_opf_case793_goc.m",
    "pglib_opf_case1354_pegase.m",
    "pglib_opf_case1803_snem.m",
    "pglib_opf_case1888_rte.m",
    "pglib_opf_case1951_rte.m",
    "pglib_opf_case2000_goc.m",
    "pglib_opf_case2312_goc.m",
    "pglib_opf_case2383wp_k.m",
    "pglib_opf_case2736sp_k.m",
    "pglib_opf_case2737sop_k.m",
    "pglib_opf_case2742_goc.m",
    "pglib_opf_case2746wop_k.m",
    "pglib_opf_case2746wp_k.m",
    "pglib_opf_case2848_rte.m",
    "pglib_opf_case2853_sdet.m",
    "pglib_opf_case2868_rte.m",
    "pglib_opf_case2869_pegase.m",
    "pglib_opf_case3012wp_k.m",
    "pglib_opf_case3022_goc.m",
    "pglib_opf_case3120sp_k.m",
    "pglib_opf_case3375wp_k.m",
    "pglib_opf_case3970_goc.m",
    "pglib_opf_case4020_goc.m",
    "pglib_opf_case4601_goc.m",
    "pglib_opf_case4619_goc.m",
    "pglib_opf_case4661_sdet.m",
    "pglib_opf_case4837_goc.m",
    "pglib_opf_case4917_goc.m",
    "pglib_opf_case5658_epigrids.m",
    "pglib_opf_case6468_rte.m",
    "pglib_opf_case6470_rte.m",
    "pglib_opf_case6495_rte.m",
    "pglib_opf_case6515_rte.m",
    "pglib_opf_case7336_epigrids.m",
    "pglib_opf_case8387_pegase.m",
    "pglib_opf_case9241_pegase.m",
    "pglib_opf_case9591_goc.m",
    "pglib_opf_case10000_goc.m",
    "pglib_opf_case10192_epigrids.m",
    "pglib_opf_case10480_goc.m",
    "pglib_opf_case13659_pegase.m",
    "pglib_opf_case19402_goc.m",
    "pglib_opf_case20758_epigrids.m",
    "pglib_opf_case24464_goc.m",
    "pglib_opf_case30000_goc.m",
    "pglib_opf_case78484_epigrids.m",
]
allparams = Dict(:case_file => case_files)
experiments = dict_list(allparams)
nexperiments = length(experiments)

for (i, d) in enumerate(experiments)
    @info "Running experiment $i/$nexperiments:" d

    # Run experiment if it doesn't exist yet, save result to file with git commit info
    res, file = produce_or_load(run_opf_benchmarks, d, datadir("results"); tag=true)
    yield()
end
