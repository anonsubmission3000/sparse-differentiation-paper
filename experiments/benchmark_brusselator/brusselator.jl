# Brusselator example taken from Gowda et al.
# "Sparsity Programming: Automated Sparsity-Aware Optimizations in Differentiable Programming"
# https://openreview.net/pdf?id=rJlPdcY38B

# Up-to-date code from SciMLBenchmarks
# https://docs.sciml.ai/SciMLBenchmarksOutput/stable/NonlinearProblem/bruss/

brusselator_f(x, y) = (((x - 3//10)^2 + (y - 6//10)^2) ≤ 0.01) * 5

limit(a, N) = ifelse(a == N + 1, 1, ifelse(a == 0, N, a))

function init_brusselator_2d(xyd, N)
    N = length(xyd)
    u = zeros(N, N, 2)
    for I in CartesianIndices((N, N))
        x = xyd[I[1]]
        y = xyd[I[2]]
        u[I, 1] = 22 * (y * (1 - y))^(3 / 2)
        u[I, 2] = 27 * (x * (1 - x))^(3 / 2)
    end
    return u
end

function brusselator_2d_loop!(du_, u_, p)
    A, B, α, δx, N = p
    α = α / δx^2
    xyd_brusselator = range(0; stop=1, length=N)

    du = reshape(du_, N, N, 2)
    u = reshape(u_, N, N, 2)

    @inbounds @simd for I in CartesianIndices((N, N))
        i, j = Tuple(I)
        x, y = xyd_brusselator[I[1]], xyd_brusselator[I[2]]
        ip1, im1 = limit(i + 1, N), limit(i - 1, N)
        jp1, jm1 = limit(j + 1, N), limit(j - 1, N)

        du[i, j, 1] =
            α * (u[im1, j, 1] + u[ip1, j, 1] + u[i, jp1, 1] + u[i, jm1, 1] - 4u[i, j, 1]) +
            B +
            u[i, j, 1]^2 * u[i, j, 2] - (A + 1) * u[i, j, 1] + brusselator_f(x, y)

        du[i, j, 2] =
            α * (u[im1, j, 2] + u[ip1, j, 2] + u[i, jp1, 2] + u[i, jm1, 2] - 4u[i, j, 2]) +
            A * u[i, j, 1] - u[i, j, 1]^2 * u[i, j, 2]
    end
    return nothing
end

struct Brusselator!{U,P}
    u0::U
    p::P
end

Base.show(b!::Brusselator!) = "Brusselator(N=$(b!.N))"

function Brusselator!(N::Integer)
    xyd_brusselator = range(0; stop=1, length=N)
    u0 = vec(init_brusselator_2d(xyd_brusselator, N))
    p = (3.4, 1.0, 10.0, step(xyd_brusselator), N)
    return Brusselator!(u0, p)
end

(b!::Brusselator!)(y, x) = brusselator_2d_loop!(y, x, b!.p)
