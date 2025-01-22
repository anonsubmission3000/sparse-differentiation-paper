#= Read README before running! =#

using ADNLPModels, NLPModels, NLPModelsIpopt
using Memento # turn off PowerModels.jl's verbosity

pm_logger = getlogger("PowerModels")
setlevel!(pm_logger, "warn")

include("rosetta-opf/nlpmodels.jl")

const case_dir = joinpath(@__DIR__, "pglib-opf")

function nlp_lagrangian(nlp::AbstractNLPModel, x::AbstractVector)
    f = NLPModels.obj(nlp, x)
    c = similar(x, nlp.meta.ncon)
    NLPModels.cons!(nlp, x, c)
    L = f + sum(c)
    return L
end

function load_opf_case(case_name::String)
    @info "Loading case..."
    path = joinpath(case_dir, case_name)

    dataset = load_and_setup_data(path)
    model = build_opf_optimization_prob(dataset)
    nlp = model.nlp

    lagrangian = Base.Fix1(nlp_lagrangian, nlp)
    var_init = nlp.meta.x0
    n_vars = nlp.meta.nvar
    n_cons = nlp.meta.ncon

    return lagrangian, var_init, n_vars, n_cons
end
