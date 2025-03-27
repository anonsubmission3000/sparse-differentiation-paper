### A Pluto.jl notebook ###
# v0.20.5

using Markdown
using InteractiveUtils

# ╔═╡ 56c116ec-0949-11f0-0637-e50b25868282
begin
    using Pkg
    Pkg.activate(@__DIR__)
    using ADTypes
    using Arpack
    using CairoMakie
    using ComponentArrays
    using DifferentiationInterface
    using Graphs
    using ImplicitDifferentiation
    using Krylov
    using LinearAlgebra
    using Lux
    using NNlib
    using Optimisers
    using PlutoUI
    using Lux, LuxCore
    using Random: AbstractRNG
    using SparseConnectivityTracer
    using SparseMatrixColorings
    using StableRNGs: StableRNG
    using StatsBase
    using Zygote: Zygote
end

# ╔═╡ 364e29e8-3fd9-40aa-bab4-36cb58030bd0
md"""
# Implicit GNNs
"""

# ╔═╡ b200f2f6-bbcb-4aaf-a851-fee098584ee1
md"""
Sources:

- <https://github.com/SwiftieH/IGNN/blob/b3c1ab31f0167827d44f1875cb8f84b725a06905/nodeclassification/>
"""

# ╔═╡ 25d13a3e-a3d9-4854-be9f-90fa9ccfcd69
TableOfContents()

# ╔═╡ 987feeca-6285-4ab4-a967-519db762d9a8
md"""
## Hyperparameters
"""

# ╔═╡ fc60021b-ae5d-43a4-a8f7-7f37d7585424
begin
    const CHAIN_LENGTH = 10
    const FEATURE_DIM = 100
    const NOISE = 0.01  # different from the original
    const TRAIN_GRAPHS_PER_CLASS = 1
    const VAL_GRAPHS_PER_CLASS = 5
    const TEST_GRAPHS_PER_CLASS = 10
    const DROPOUT = 0.5
    const KAPPA = 0.9
    const LEARNING_RATE = 1e-2
    const WEIGHT_DECAY = 5e-4
    const EPOCHS = 100
    const HIDDEN = 16
    const MAX_ITERATIONS = 300
    const CONVERGENCE_TOL = 3e-6
end;

# ╔═╡ ee2b1dce-7ecd-4a2e-9e21-0434138f08a9
function MyAutoSparse(backend)
    return AutoSparse(
        backend;
        sparsity_detector=TracerSparsityDetector(),
        coloring_algorithm=GreedyColoringAlgorithm(),
    )
end

# ╔═╡ 44b9bf3a-7f63-4207-9938-665fe7fe26f3
md"""
## Chains dataset
"""

# ╔═╡ a6ae1eb9-80cb-4d5c-a34e-6e50337e216f
function sample_single_chain(
    rng::AbstractRNG, class::Bool; chain_length, feature_dim, noise::R
) where {R<:Real}
    U = noise .* (rand(rng, R, feature_dim, chain_length) .- R(0.5))
    U[1, 1] = class
    y = zeros(Bool, 2, chain_length)
    y[class + 1, :] .= true
    return (; U, y)
end

# ╔═╡ fd587417-3f57-4c5b-afe5-455bce2e4fcd
function sample_chains_dataset(
    rng::AbstractRNG; chain_length, feature_dim, noise, graphs_per_class
)
    data = vcat(
        [
            sample_single_chain(rng, false; chain_length, feature_dim, noise) for
            _ in 1:graphs_per_class
        ],
        [
            sample_single_chain(rng, true; chain_length, feature_dim, noise) for
            _ in 1:graphs_per_class
        ],
    )
    return data
end

# ╔═╡ 3e8f0fcd-c2ae-46f6-bb1c-840c990b7a05
function sample_full_chains_dataset(
    rng::AbstractRNG;
    chain_length,
    feature_dim,
    noise,
    train_graphs_per_class,
    test_graphs_per_class,
    val_graphs_per_class,
)
    train = sample_chains_dataset(
        rng; chain_length, feature_dim, noise, graphs_per_class=train_graphs_per_class
    )
    val = sample_chains_dataset(
        rng; chain_length, feature_dim, noise, graphs_per_class=val_graphs_per_class
    )
    test = sample_chains_dataset(
        rng; chain_length, feature_dim, noise, graphs_per_class=test_graphs_per_class
    )
    return (; train, val, test)
end

# ╔═╡ 3a24a53c-8274-442b-a908-9815583d5a6e
md"""
## Architecture
"""

# ╔═╡ e57e54cb-56df-4708-b476-31ecd34aaddc
function normalize_adjacency(A)
    Ã = A + I
    D̃ = Diagonal(dropdims(sum(Ã; dims=1); dims=1))
    D̃_sqrt_inv = inv(sqrt(D̃))
    return D̃_sqrt_inv * Ã * D̃_sqrt_inv
end

# ╔═╡ 659443d3-1955-42fe-861e-4e6bdac9e69f
spectral_radius(A) = abs(only(first(Arpack.eigs(A; nev=1, which=:LM))))

# ╔═╡ dfcc5572-afc3-410c-93b6-1a771b3833eb
begin
    @kwdef struct UnrolledGNNLayer{R<:Real,M<:AbstractMatrix,F} <: LuxCore.AbstractLuxLayer
        n::Int  # number of nodes
        p::Int  # input dim
        m::Int  # output dim
        κ::R  # norm scaling
        max_iterations::Int
        convergence_tol::Float64
        ϕ::F = relu   # activation function
        A::M = typeof(κ).(normalize_adjacency(adjacency_matrix(path_graph(n))))  # adjacency matrix
        λ::R = spectral_radius(A) # PF eigenvalue
    end

    function LuxCore.initialparameters(rng::AbstractRNG, l::UnrolledGNNLayer{R}) where {R}
        (; p, m) = l
        stdev = R(1 / sqrt(m))
        return (;
            W=stdev .* (rand(rng, R, m, m) .- R(0.5)),
            Ω=stdev .* (rand(rng, R, m, p) .- R(0.5)),
        )
    end

    LuxCore.initialstates(::AbstractRNG, ::UnrolledGNNLayer) = NamedTuple()

    function (l::UnrolledGNNLayer)(U, ps, st::NamedTuple)
        (; A, λ, n, p, m, ϕ, κ, max_iterations, convergence_tol) = l
        (; W, Ω) = ps
        X = zeros(eltype(U), m, n)
        W_proj = (W ./ opnorm(W, Inf)) .* (κ / λ)
        for _ in 1:max_iterations
            X_new = ϕ.(W_proj * X * A + Ω * U)
            if norm(X - X_new, Inf) < convergence_tol
                return X_new, st
            else
                X = X_new
            end
        end
        return X, st
    end
end

# ╔═╡ 7cbe698b-0a31-4311-9f04-f793fc65910e
function solver(ps, unrolled_gnn_layer, U, st)
    X, _ = unrolled_gnn_layer(U, ps, st)
    X_vec = vec(X)
    return X_vec, nothing
end

# ╔═╡ d723af47-1d9c-44fc-9153-0e0696202a05
function conditions(ps, X_vec, _byproduct, unrolled_gnn_layer, U, st)
    (; A, λ, n, p, m, ϕ, κ) = unrolled_gnn_layer
    (; W, Ω) = ps
    X = reshape(X_vec, m, n)
    W_proj = (W ./ opnorm(W, Inf)) .* (κ / λ)
    X_new = ϕ.(W_proj * X * A + Ω * U)
    return vec(X_new) - X_vec
end

# ╔═╡ 3f50d5da-2949-4cad-ba52-306aa1ca8596
begin
    @kwdef struct ImplicitGNNLayer{U,I} <: LuxCore.AbstractLuxLayer
        unrolled::U
        implicit::I
    end

    function LuxCore.initialparameters(rng::AbstractRNG, l::ImplicitGNNLayer{R}) where {R}
        return LuxCore.initialparameters(rng, l.unrolled)
    end

    LuxCore.initialstates(::AbstractRNG, ::ImplicitGNNLayer) = NamedTuple()

    function (l::ImplicitGNNLayer)(U, ps, st::NamedTuple)
        (; implicit, unrolled) = l
        (; n, m) = unrolled
        X_vec, _ = implicit(ps, unrolled, U, st)
        return reshape(X_vec, m, n), st
    end
end

# ╔═╡ fd2272e2-80fe-4ef6-a3e8-7b7643ccd117
md"""
## Experiment
"""

# ╔═╡ 325fe70a-ec28-4746-9306-a720d40b0659
begin
    function compute_loss_curve(rng; layer_type=:iterative)
        data = sample_full_chains_dataset(
            rng;
            chain_length=CHAIN_LENGTH,
            feature_dim=FEATURE_DIM,
            noise=NOISE,
            train_graphs_per_class=TRAIN_GRAPHS_PER_CLASS,
            val_graphs_per_class=VAL_GRAPHS_PER_CLASS,
            test_graphs_per_class=TEST_GRAPHS_PER_CLASS,
        )

        prep_time = @elapsed begin
            unrolled_gnn_layer = UnrolledGNNLayer(;
                n=CHAIN_LENGTH,
                p=FEATURE_DIM,
                m=HIDDEN,
                κ=KAPPA,
                max_iterations=MAX_ITERATIONS,
                convergence_tol=CONVERGENCE_TOL,
            )

            input_example = let
                ps, st = Lux.setup(StableRNG(0), unrolled_gnn_layer)
                ps = ComponentVector(ps)
                U = data.train[1].U
                (ps, unrolled_gnn_layer, U, st)
            end

            if layer_type == :iterative
                layer = ImplicitGNNLayer(;
                    unrolled=unrolled_gnn_layer,
                    implicit=ImplicitFunction(
                        solver,
                        conditions;
                        representation=OperatorRepresentation(),
                        backend=AutoZygote(),
                    ),
                )
            else
                layer = ImplicitGNNLayer(;
                    unrolled=unrolled_gnn_layer,
                    implicit=ImplicitFunction(
                        solver,
                        conditions;
                        representation=MatrixRepresentation(),
                        linear_solver=\,
                        backend=MyAutoSparse(AutoForwardDiff()),
                        preparation=ADTypes.ReverseMode(),
                        input_example=input_example,
                    ),
                )
            end

            model = Chain(layer, Dropout(DROPOUT), Dense(HIDDEN, 2; use_bias=false))

            optimiser = OptimiserChain(WeightDecay(WEIGHT_DECAY), Adam(LEARNING_RATE))

            ps, st = Lux.setup(rng, model)
            ps = ComponentVector(ps)
            train_state = Lux.Training.TrainState(model, ps, st, optimiser)
        end

        times = Float64[]
        losses = Float64[]
        accuracies = Float64[]

        local loss
        for epoch in 1:EPOCHS
            # apply gradient step
            time = @elapsed for (k, (; U, y)) in enumerate(data.train)
                gs, loss, stats, train_state = Lux.Training.single_train_step(
                    AutoZygote(), BinaryCrossEntropyLoss(; logits=true), (U, y), train_state
                )
            end
            push!(times, time)
            push!(losses, loss)

            # evaluate test accuracy
            ps, st = train_state.parameters, train_state.states
            nb_errors = 0
            for (; U, y) in data.test
                y_model = Lux.apply(model, U, ps, Lux.testmode(st))[1]
                predicted_classes = getindex.(argmax(y_model; dims=1), 1)
                true_classes = getindex.(argmax(y; dims=1), 1)
                nb_errors += sum(predicted_classes .!= true_classes)
            end
            error_rate = nb_errors / (length(data.test) * length(data.test[1].y))
            accuracy = 1 - error_rate
            push!(accuracies, accuracy)
        end

        return times, losses, accuracies, prep_time
    end

    # first run for compilation
    compute_loss_curve(StableRNG(0); layer_type=:iterative)
    compute_loss_curve(StableRNG(0); layer_type=:direct)
end

# ╔═╡ 4cab275f-ceb6-48a8-9298-6ec7a2ee616d
let
    # second run for real measurements
    times1, losses1, acc1, prep_time1 = compute_loss_curve(
        StableRNG(0); layer_type=:iterative
    )
    times2, losses2, acc2, prep_time2 = compute_loss_curve(StableRNG(0); layer_type=:direct)

    with_theme(theme_latexfonts()) do
        fig = Figure()
        ax1 = CairoMakie.Axis(fig[1, 1]; ylabel="Training loss")
        ax2 = CairoMakie.Axis(fig[2, 1]; xlabel="Wall time [s]", ylabel="Test accuracy")

        scatterlines!(
            ax1,
            prep_time1 .+ cumsum(times1),
            losses1;
            label="VJP (iterative solve)",
            marker=:xcross,
            markersize=7,
        )
        scatterlines!(
            ax1,
            prep_time2 .+ cumsum(times2),
            losses2;
            label="Sparse Jacobian (direct solve)",
            marker=:circle,
            markersize=7,
        )

        scatterlines!(ax2, prep_time1 .+ cumsum(times1), acc1; marker=:xcross, markersize=7)
        scatterlines!(ax2, prep_time2 .+ cumsum(times2), acc2; marker=:circle, markersize=7)

        axislegend(ax1; position=:rt)
        #axislegend(ax2; position=:rb)
        save(joinpath(@__DIR__, "plots", "training_speedup_ignn.png"), fig)
        fig
    end
end

# ╔═╡ Cell order:
# ╟─364e29e8-3fd9-40aa-bab4-36cb58030bd0
# ╟─b200f2f6-bbcb-4aaf-a851-fee098584ee1
# ╠═56c116ec-0949-11f0-0637-e50b25868282
# ╠═25d13a3e-a3d9-4854-be9f-90fa9ccfcd69
# ╟─987feeca-6285-4ab4-a967-519db762d9a8
# ╠═fc60021b-ae5d-43a4-a8f7-7f37d7585424
# ╠═ee2b1dce-7ecd-4a2e-9e21-0434138f08a9
# ╟─44b9bf3a-7f63-4207-9938-665fe7fe26f3
# ╠═a6ae1eb9-80cb-4d5c-a34e-6e50337e216f
# ╠═fd587417-3f57-4c5b-afe5-455bce2e4fcd
# ╠═3e8f0fcd-c2ae-46f6-bb1c-840c990b7a05
# ╟─3a24a53c-8274-442b-a908-9815583d5a6e
# ╠═e57e54cb-56df-4708-b476-31ecd34aaddc
# ╠═659443d3-1955-42fe-861e-4e6bdac9e69f
# ╠═dfcc5572-afc3-410c-93b6-1a771b3833eb
# ╠═7cbe698b-0a31-4311-9f04-f793fc65910e
# ╠═d723af47-1d9c-44fc-9153-0e0696202a05
# ╠═3f50d5da-2949-4cad-ba52-306aa1ca8596
# ╟─fd2272e2-80fe-4ef6-a3e8-7b7643ccd117
# ╠═325fe70a-ec28-4746-9306-a720d40b0659
# ╠═4cab275f-ceb6-48a8-9298-6ec7a2ee616d
